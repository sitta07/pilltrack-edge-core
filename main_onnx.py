#!/usr/bin/env python3
"""
PILLTRACK – HYBRID ENGINE (YOLO ONNX + DINOv2 PYTORCH)
✔ YOLO runs via ONNX Runtime for speed on Pi
✔ DINOv2 runs via PyTorch (Batch Mode)
✔ 10 FPS Display Throttling
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
import onnxruntime as ort 

from dataclasses import dataclass, field
from typing import Tuple, List, Dict
from sync_manager import SyncManager

# ================= ⚙️ CONFIG =================
with open("config.yaml", "r") as f:
    yaml_cfg = yaml.safe_load(f)

@dataclass
class Config:
    # ต้องมั่นใจว่า path นี้ชี้ไปที่ไฟล์ .onnx
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

# ================= 🛠️ TEXT =================
def draw_text(img, text, pos, scale=0.55, color=(255,255,255), thickness=2):
    cv2.putText(img, text, pos, FONT, scale, (0,0,0), thickness+3)
    cv2.putText(img, text, pos, FONT, scale, color, thickness)

# ================= 🧠 NAME NORMALIZATION =================
def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'_pack.*', '', name)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

# ================= 🧠 RX MANAGER =================
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
                    print(f"🎉 VERIFIED: {original.upper()}")
                    self.verified.add(original)
                return True
        return False

# ================= 🎨 FEATURE ENGINE =================
class FeatureEngine:
    def __init__(self):
        print("⏳ Loading DINOv2 (ViT-B/14)...")
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

# ================= 🤖 AI PROCESSOR (ONNX VERSION) =================
class AIProcessor:
    def __init__(self):
        self.rx = PrescriptionManager()
        self.engine = FeatureEngine()
        
        # 1. Initialize YOLO ONNX
        print(f"⏳ Loading YOLO ONNX from {CFG.MODEL_PACK}...")
        # ใช้ CPU Execution Provider สำหรับ Raspberry Pi
        self.ort_session = ort.InferenceSession(CFG.MODEL_PACK, providers=['CPUExecutionProvider'])
        self.input_name = self.ort_session.get_inputs()[0].name
        
        self.db_vectors = []
        self.db_names = []
        self.db_sift_map = {}
        self.bf = cv2.BFMatcher()
        self.load_db()
        
        self.latest = None
        self.lock = threading.Lock()
        self.ms = 0
        self.last_log_time = 0 

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
                if score > max_score: max_score = score
            except: continue
        return max_score

    def process(self, frame):
        t0 = time.time()
        
        # 1. YOLO ONNX Inference
        img_resized = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        # Preprocess: HWC -> NCHW + Normalize
        blob = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        
        outputs = self.ort_session.run(None, {self.input_name: blob})
        
        # YOLOv8 ONNX output shape is (1, boxes+classes, 3549) - Need post-processing
        # สำหรับความกระชับ ในที่นี้เราจะใช้ Logic การกรองแบบง่าย หรือถ้าใช้ YOLOv10/v11 จะง่ายกว่า
        # หมายเหตุ: โค้ดส่วนนี้ assume ว่าเป็นโครงสร้าง output มาตรฐานของ YOLOv8 ONNX
        output = np.squeeze(outputs[0])
        output = output.transpose() # (3549, boxes+classes)
        
        boxes = []
        scores = []
        class_ids = []
        
        for row in output:
            prob = row[4:].max()
            if prob > CFG.CONF_THRESHOLD:
                scores.append(prob)
                boxes.append(row[:4]) # cx, cy, w, h
                class_ids.append(row[4:].argmax())

        if not boxes:
            self.ms = (time.time()-t0)*1000
            return

        # 2. Collect Crops
        crops = []
        sx, sy = CFG.DISPLAY_SIZE[0]/CFG.AI_SIZE, CFG.DISPLAY_SIZE[1]/CFG.AI_SIZE

        for box in boxes:
            cx, cy, w, h = box
            x1, y1 = int((cx - w/2) * sx), int((cy - h/2) * sy)
            x2, y2 = int((cx + w/2) * sx), int((cy + h/2) * sy)
            
            crop = frame[max(0,y1):min(CFG.DISPLAY_SIZE[1],y2), 
                         max(0,x1):min(CFG.DISPLAY_SIZE[0],x2)]
            if crop.size > 0 and crop.shape[0] > 10:
                crops.append(crop)

        if not crops: return

        # 3. DINO & Matrix Match (เหมือนเดิม)
        batch_dino_vecs = self.engine.extract_dino_batch(crops)
        similarity_matrix = np.dot(batch_dino_vecs, self.db_vectors.T)

        for i, crop in enumerate(crops):
            sim_scores = similarity_matrix[i]
            top_k_indices = np.argsort(sim_scores)[::-1][:5]
            
            best_final_score, best_label = -1.0, "Unknown"
            q_des = self.engine.extract_sift(crop)
            seen_names = set()
            
            for idx in top_k_indices:
                candidate_name = self.db_names[idx]
                if candidate_name in seen_names: continue
                seen_names.add(candidate_name)
                
                dino_score = sim_scores[idx]
                if dino_score < 0.4: continue

                sift_score = self.get_sift_score(q_des, self.db_sift_map.get(candidate_name, []))
                final_score = (dino_score * CFG.W_DINO) + (sift_score * CFG.W_SIFT)

                if final_score > best_final_score:
                    best_final_score, best_label = final_score, candidate_name

            if best_final_score > 0.60:
                self.rx.verify(best_label)

        self.ms = (time.time()-t0)*1000

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def loop(self):
        while True:
            if self.latest is not None:
                with self.lock:
                    frame_to_process = self.latest.copy()
                self.process(frame_to_process)
            time.sleep(0.005)

# ================= 📷 CAMERA =================
class Camera:
    def __init__(self):
        try:
            from picamera2 import Picamera2
            self.cam = Picamera2()
            self.cam.configure(self.cam.create_preview_configuration(main={"size":CFG.DISPLAY_SIZE,"format":"RGB888"}))
            self.cam.start()
            self.pi = True
        except:
            self.cap = cv2.VideoCapture(0)
            self.pi = False

    def get(self):
        if self.pi: return self.cam.capture_array()
        ret, f = self.cap.read()
        return f if ret else None

# ================= 🖥️ UI =================
def draw_ui(frame, rx):
    right_margin, y = frame.shape[1] - 10, 40
    for drug in rx.all_drugs:
        done = drug in rx.verified
        color = (0,255,0) if done else (180,180,180)
        text = f"✔ {drug.upper()}" if done else f"□ {drug.upper()}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 2)
        draw_text(frame, text, (right_margin - tw, y), 0.55, color)
        if done: cv2.line(frame, (right_margin-tw, y-8), (right_margin, y-8), (0,255,0), 2)
        y += 28

# ================= 🚀 MAIN =================
if __name__ == "__main__":
    try: SyncManager().sync()
    except: pass

    cam = Camera()
    ai = AIProcessor().start()
    cv2.namedWindow("PillTrack", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PillTrack", *CFG.DISPLAY_SIZE)

    DISPLAY_FPS = 10
    display_interval = 1.0 / DISPLAY_FPS
    last_display_time = 0

    while True:
        frame = cam.get()
        if frame is None: continue

        with ai.lock: ai.latest = frame
        
        current_time = time.time()
        if current_time - last_display_time > display_interval:
            display_frame = frame.copy()
            draw_ui(display_frame, ai.rx)
            cv2.putText(display_frame, f"AI (ONNX): {ai.ms:.1f}ms", (10, 20), FONT, 0.5, (0,255,255), 1)
            cv2.imshow("PillTrack", display_frame)
            last_display_time = current_time
            if cv2.waitKey(1) == ord('q'): break
        else:
            time.sleep(0.001)
            
    cv2.destroyAllWindows()