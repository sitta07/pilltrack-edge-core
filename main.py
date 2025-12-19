#!/usr/bin/env python3
"""
PILLTRACK – SENIOR EDITION (HYBRID ENGINE + LIVE VISUALIZATION)
✔ Real-time Bounding Boxes & Labels
✔ Thread-safe shared results
✔ Matrix Vectorized Search + SIFT Refinement
✔ 10FPS UI / Full-speed AI Inference
"""

import os
import re
import time
import threading
import yaml
import json
import pickle
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Tuple, List, Dict
from ultralytics import YOLO
from sync_manager import SyncManager

# ================= ⚙️ CONFIG =================
with open("config.yaml", "r") as f:
    yaml_cfg = yaml.safe_load(f)

@dataclass
class Config:
    MODEL_PACK: str = yaml_cfg['artifacts']['model']
    DB_PACKS_VEC: str = "database/pill_fingerprints.pkl"
    DRUG_LIST_JSON: str = yaml_cfg['artifacts']['drug_list']
    DISPLAY_SIZE: Tuple[int, int] = (
        yaml_cfg['display']['width'],
        yaml_cfg['display']['height']
    )
    AI_SIZE: int = 416
    CONF_THRESHOLD: float = yaml_cfg['settings']['yolo_conf']
    
    W_DINO: float = 0.6
    W_SIFT: float = 0.4
    SIFT_SATURATION: int = 400
    
    MEAN: np.ndarray = field(default_factory=lambda: np.array([0.485, 0.456, 0.406], dtype=np.float32))
    STD: np.ndarray = field(default_factory=lambda: np.array([0.229, 0.224, 0.225], dtype=np.float32))

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ================= 🛠️ UTILS =================
def draw_text(img, text, pos, scale=0.5, color=(255,255,255), thickness=1):
    cv2.putText(img, text, pos, FONT, scale, (0,0,0), thickness+2)
    cv2.putText(img, text, pos, FONT, scale, color, thickness)

def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'_pack.*', '', name)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

# ================= 🧠 ENGINE =================
class PrescriptionManager:
    def __init__(self):
        self.all_drugs = []
        self.norm_map = {}
        self.verified = set()
        self.load()

    def load(self):
        if not os.path.exists(CFG.DRUG_LIST_JSON): return
        with open(CFG.DRUG_LIST_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for d in data.get('drugs', []):
            norm = normalize_name(d)
            self.all_drugs.append(d.lower())
            self.norm_map[norm] = d.lower()

    def verify(self, detected_name: str):
        norm_det = normalize_name(detected_name)
        for norm_drug, original in self.norm_map.items():
            if norm_det.startswith(norm_drug) or norm_drug.startswith(norm_det):
                if original not in self.verified:
                    print(f"🔥 [NEW VERIFIED]: {original.upper()}")
                    self.verified.add(original)
                return True
        return False

class FeatureEngine:
    def __init__(self):
        print("⏳ Loading DINOv2...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval().to(device)
        self.sift = cv2.SIFT_create()

    def preprocess_batch(self, crop_list):
        batch = []
        for img in crop_list:
            img = cv2.resize(img, (224, 224))
            img = (img.astype(np.float32) / 255.0 - CFG.MEAN) / CFG.STD
            img = img.transpose(2, 0, 1)
            batch.append(img)
        return np.array(batch)

    @torch.no_grad()
    def extract_dino_batch(self, crop_list):
        if not crop_list: return np.array([])
        img_batch_np = self.preprocess_batch(crop_list)
        img_batch_t = torch.from_numpy(img_batch_np).to(device)
        embeddings = self.model(img_batch_t)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def extract_sift(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, sift_des = self.sift.detectAndCompute(gray, None)
        return sift_des

# ================= 🤖 AI PROCESSOR =================
class AIProcessor:
    def __init__(self):
        self.rx = PrescriptionManager()
        self.engine = FeatureEngine()
        self.db_vectors = []
        self.db_names = []
        self.db_sift_map = {}
        self.bf = cv2.BFMatcher()
        self.load_db()
        
        print("⏳ Loading YOLO...")
        self.yolo = YOLO(CFG.MODEL_PACK)
        
        self.latest_frame = None
        self.results = []  # Stores: [{'box': [x1,y1,x2,y2], 'label': str, 'conf': float}]
        self.lock = threading.Lock()
        self.ms = 0

    def load_db(self):
        if not os.path.exists(CFG.DB_PACKS_VEC): return
        with open(CFG.DB_PACKS_VEC,'rb') as f:
            raw = pickle.load(f)
        for name, data in raw.items():
            dino_list = data.get('dino', []) if isinstance(data, dict) else data
            self.db_sift_map[name] = data.get('sift', []) if isinstance(data, dict) else []
            for vec in dino_list:
                self.db_vectors.append(np.array(vec))
                self.db_names.append(name)
        self.db_vectors = np.array(self.db_vectors)
        norms = np.linalg.norm(self.db_vectors, axis=1, keepdims=True)
        self.db_vectors = self.db_vectors / (norms + 1e-8)

    def get_sift_score(self, query_des, target_des_list):
        if query_des is None or not target_des_list: return 0.0
        max_score = 0.0
        for target_des in target_des_list[:5]:
            try:
                matches = self.bf.knnMatch(query_des, target_des, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                score = min(len(good) / CFG.SIFT_SATURATION, 1.0)
                max_score = max(max_score, score)
            except: continue
        return max_score

    def process(self, frame):
        t0 = time.time()
        img_resized = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        res = self.yolo(img_resized, conf=CFG.CONF_THRESHOLD, verbose=False)[0]

        temp_results = []
        if res.boxes is not None:
            sx = CFG.DISPLAY_SIZE[0]/CFG.AI_SIZE
            sy = CFG.DISPLAY_SIZE[1]/CFG.AI_SIZE
            
            crops = []
            box_coords = []
            
            for box in res.boxes:
                x1,y1,x2,y2 = box.xyxy[0].int().tolist()
                # Scale to Display Size
                dx1, dy1, dx2, dy2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
                
                crop = frame[max(0, dy1):min(frame.shape[0], dy2), max(0, dx1):min(frame.shape[1], dx2)]
                if crop.size == 0: continue
                
                crops.append(crop)
                box_coords.append([dx1, dy1, dx2, dy2])

            if crops:
                batch_dino = self.engine.extract_dino_batch(crops)
                sim_matrix = np.dot(batch_dino, self.db_vectors.T)

                for i, crop in enumerate(crops):
                    sim_scores = sim_matrix[i]
                    top_k = np.argsort(sim_scores)[::-1][:5]
                    
                    q_des = self.engine.extract_sift(crop)
                    best_label, max_f = "Unknown", 0.0
                    
                    seen = set()
                    for idx in top_k:
                        name = self.db_names[idx]
                        if name in seen: continue
                        seen.add(name)
                        
                        d_score = sim_scores[idx]
                        if d_score < 0.4: continue
                        
                        s_score = self.get_sift_score(q_des, self.db_sift_map.get(name, []))
                        f_score = (d_score * CFG.W_DINO) + (s_score * CFG.W_SIFT)
                        
                        if f_score > max_f:
                            max_f = f_score
                            best_label = name

                    temp_results.append({'box': box_coords[i], 'label': best_label, 'conf': max_f})
                    if max_f > 0.6: self.rx.verify(best_label)
                    
                    # Log ตลอดเวลา (Continuous Logging)
                    print(f"📡 Tracking: {best_label} | Score: {max_f:.2f}")

        with self.lock:
            self.results = temp_results
            self.ms = (time.time()-t0)*1000

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def loop(self):
        while True:
            if self.latest_frame is not None:
                with self.lock:
                    work_frame = self.latest_frame.copy()
                self.process(work_frame)
            time.sleep(0.01)

# ================= 🖥️ UI & DISPLAY =================
def draw_ui(frame, ai_proc):
    # 1. Draw Checklist
    rx = ai_proc.rx
    y = 40
    for drug in rx.all_drugs:
        done = drug in rx.verified
        color = (0,255,0) if done else (180,180,180)
        text = f"✔ {drug.upper()}" if done else f"□ {drug.upper()}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 2)
        draw_text(frame, text, (CFG.DISPLAY_SIZE[0] - tw - 10, y), 0.55, color)
        if done: cv2.line(frame, (CFG.DISPLAY_SIZE[0]-tw-10, y-8), (CFG.DISPLAY_SIZE[0]-10, y-8), (0,255,0), 2)
        y += 28

    # 2. Draw Detections (Bounding Boxes)
    with ai_proc.lock:
        current_results = ai_proc.results
        latency = ai_proc.ms

    for res in current_results:
        x1, y1, x2, y2 = res['box']
        lbl = res['label'].upper()
        cf = res['conf']
        
        # Color based on confidence
        color = (0, 255, 0) if cf > 0.6 else (0, 255, 255)
        
        # Draw Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label Tag
        tag = f"{lbl} {cf:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.4, 1)
        cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), color, -1)
        cv2.putText(frame, tag, (x1, y1-5), FONT, 0.4, (0,0,0), 1)

    # Status Info
    draw_text(frame, f"AI: {latency:.1f}ms", (10, 20), 0.5, (0,255,255))

# ================= 🚀 RUN =================
if __name__ == "__main__":
    try: SyncManager().sync()
    except: pass

    # Initialize Camera
    try:
        from picamera2 import Picamera2
        cam_obj = Picamera2()
        cam_obj.configure(cam_obj.create_preview_configuration(main={"size":CFG.DISPLAY_SIZE,"format":"RGB888"}))
        cam_obj.start()
        def get_frame(): return cam_obj.capture_array()
    except:
        cap = cv2.VideoCapture(0)
        def get_frame(): 
            ret, f = cap.read()
            return f if ret else None

    ai = AIProcessor().start()
    cv2.namedWindow("PillTrack", cv2.WINDOW_NORMAL)

    last_ui_time = 0
    while True:
        frame = get_frame()
        if frame is None: continue

        # Send to AI Thread
        with ai.lock:
            ai.latest_frame = frame

        # UI Update (10 FPS)
        if time.time() - last_ui_time > 0.1:
            display_f = frame.copy()
            draw_ui(display_f, ai)
            cv2.imshow("PillTrack", display_f)
            last_ui_time = time.time()

        if cv2.waitKey(1) == ord('q'): break

    cv2.destroyAllWindows()