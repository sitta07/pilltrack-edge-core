#!/usr/bin/env python3
"""
PILLTRACK – SENIOR EDITION (ULTRA OPTIMIZED)
✔ Model: DINOv2 ViT-Small (Fast for CPU)
✔ Logic: Lazy SIFT + Early Skip
✔ Bug Fix: Handled ragged SIFT descriptors & fixed sequence errors
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
import faiss
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
from ultralytics import YOLO
from collections import deque

# ================= ⚙️ CONFIG =================
with open("config.yaml", "r") as f:
    yaml_cfg = yaml.safe_load(f)

@dataclass
class Config:
    MODEL_PACK: str = yaml_cfg['artifacts']['model']
    DB_PACKS_VEC: str = "database/pill_fingerprints.pkl"
    DRUG_LIST_JSON: str = yaml_cfg['artifacts']['drug_list']
    DISPLAY_SIZE: Tuple[int, int] = (yaml_cfg['display']['width'], yaml_cfg['display']['height'])
    AI_SIZE: int = 416
    CONF_THRESHOLD: float = yaml_cfg['settings']['yolo_conf']
    
    # Weights & Thresholds
    W_DINO: float = 0.6
    W_SIFT: float = 0.4
    MIN_DINO_SKIP: float = 0.5   # ต่ำกว่านี้ไม่ทำ SIFT
    MIN_DINO_TOTAL: float = 0.3  # ต่ำกว่านี้ไม่เอาเข้าระบบเลย
    VERIFY_THRESHOLD: float = 0.6
    SIFT_SATURATION: int = 300   # ลดลงเพื่อให้คะแนนพุ่งง่ายขึ้น
    
    # Performance
    AI_FRAME_SKIP: int = 3       # ข้ามเฟรมมากขึ้นเพื่อ FPS
    SIFT_FEATURES: int = 200     # ลดจุด SIFT เพื่อความเร็ว
    DINO_TOP_K: int = 5
    
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
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

# ================= 🧠 PRESCRIPTION MANAGER =================
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

    def verify(self, detected_name: str) -> bool:
        norm_det = normalize_name(detected_name)
        for norm_drug, original in self.norm_map.items():
            if norm_det.startswith(norm_drug) or norm_drug.startswith(norm_det):
                if original not in self.verified:
                    print(f"🔥 [VERIFIED]: {original.upper()}")
                    self.verified.add(original)
                return True
        return False

# ================= 🔍 FEATURE ENGINE =================
class FeatureEngine:
    def __init__(self):
        print("⏳ Loading DINOv2-Small (Optimized for FPS)...")
        # เปลี่ยนเป็น vits14 เพื่อให้รันบน CPU ได้เร็วขึ้น
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.eval().to(device)
        if device.type == 'cuda': self.model = self.model.half()
        
        self.sift = cv2.SIFT_create(nfeatures=CFG.SIFT_FEATURES)
        # FLANN Matcher เร็วกว่า BFMatcher สำหรับ SIFT
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def preprocess_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        batch = np.zeros((len(crop_list), 3, 224, 224), dtype=np.float32)
        for i, img in enumerate(crop_list):
            img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            img_norm = (img_resized.astype(np.float32) / 255.0 - CFG.MEAN) / CFG.STD
            batch[i] = img_norm.transpose(2, 0, 1)
        return batch

    @torch.no_grad()
    def extract_dino_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        if not crop_list: return np.array([])
        img_batch_t = torch.from_numpy(self.preprocess_batch(crop_list)).to(device)
        if device.type == 'cuda': img_batch_t = img_batch_t.half()
        embeddings = self.model(img_batch_t)
        return F.normalize(embeddings, p=2, dim=1).cpu().float().numpy()

    def extract_sift(self, img: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors

# ================= 🤖 AI PROCESSOR =================
class AIProcessor:
    def __init__(self):
        self.rx = PrescriptionManager()
        self.engine = FeatureEngine()
        self.db_names = []
        self.db_sift_map = {}
        self.load_db()
        self.yolo = YOLO(CFG.MODEL_PACK)
        
        self.latest_frame = None
        self.results = []
        self.lock = threading.Lock()
        self.ms = 0
        self.fps_history = deque(maxlen=30)
        self.process_counter = 0

    def load_db(self):
        if not os.path.exists(CFG.DB_PACKS_VEC): return
        with open(CFG.DB_PACKS_VEC, 'rb') as f:
            raw = pickle.load(f)
        
        vectors = []
        for name, data in raw.items():
            # ดึงเฉพาะ Dino Vector ตัวแรก (เพื่อให้ shape คงที่)
            dino_list = data.get('dino', []) if isinstance(data, dict) else data
            if len(dino_list) > 0:
                vectors.append(np.array(dino_list[0], dtype=np.float32))
                self.db_names.append(name)
                # เก็บ SIFT แยกไว้ (เพราะความยาวไม่เท่ากัน ห้ามยัดใส่ Matrix)
                self.db_sift_map[name] = data.get('sift', []) if isinstance(data, dict) else []

        vectors = np.array(vectors)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        print(f"✅ FAISS DB Ready: {len(vectors)} items")

    def get_sift_score(self, query_des: Optional[np.ndarray], target_des_list: List[np.ndarray]) -> float:
        if query_des is None or query_des.shape[0] < 2 or not target_des_list: return 0.0
        best_s = 0.0
        for target_des in target_des_list:
            if target_des is None or target_des.shape[0] < 2: continue
            try:
                # แก้ปัญหา dtype ให้ FLANN
                matches = self.engine.flann.knnMatch(query_des.astype(np.float32), 
                                                   target_des.astype(np.float32), k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                best_s = max(best_s, min(len(good) / CFG.SIFT_SATURATION, 1.0))
            except: continue
        return best_s

    def process(self, frame: np.ndarray):
        t0 = time.time()
        img_resized = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        res = self.yolo(img_resized, conf=CFG.CONF_THRESHOLD, verbose=False)[0]

        if not res.boxes:
            with self.lock: self.results = []
            return

        temp_results = []
        sx, sy = CFG.DISPLAY_SIZE[0]/CFG.AI_SIZE, CFG.DISPLAY_SIZE[1]/CFG.AI_SIZE
        crops, box_coords = [], []

        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            dx1, dy1, dx2, dy2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
            crop = frame[max(0, dy1):min(frame.shape[0], dy2), max(0, dx1):min(frame.shape[1], dx2)]
            if crop.size > 0:
                crops.append(crop)
                box_coords.append([dx1, dy1, dx2, dy2])

        if crops:
            batch_dino = self.engine.extract_dino_batch(crops)
            scores, indices = self.index.search(batch_dino, k=CFG.DINO_TOP_K)
            
            for i, crop in enumerate(crops):
                sim_scores = scores[i]
                top_idx = indices[i]
                
                # --- [FIXED: EARLY SKIP LOGIC] ---
                if np.max(sim_scores) < CFG.MIN_DINO_TOTAL: continue

                best_label, max_fusion, q_des = "Unknown", 0.0, None
                seen = set()

                for idx_k, db_idx in enumerate(top_idx):
                    if db_idx == -1: continue
                    name = self.db_names[db_idx]
                    if name in seen: continue
                    seen.add(name)

                    dino_score = sim_scores[idx_k]
                    sift_score = 0.0
                    
                    # ทำ SIFT เฉพาะตัวที่ DINO ค่อนข้างมั่นใจ (Lazy Extraction)
                    if dino_score > CFG.MIN_DINO_SKIP:
                        if q_des is None: q_des = self.engine.extract_sift(crop)
                        sift_score = self.get_sift_score(q_des, self.db_sift_map.get(name, []))
                    
                    fusion = (dino_score * CFG.W_DINO) + (sift_score * CFG.W_SIFT)
                    if fusion > max_fusion:
                        max_fusion, best_label = fusion, name

                temp_results.append({'box': box_coords[i], 'label': best_label, 'conf': max_fusion})
                if max_fusion > CFG.VERIFY_THRESHOLD: self.rx.verify(best_label)

        elapsed = (time.time() - t0) * 1000
        with self.lock:
            self.results = temp_results
            self.ms = elapsed
            self.fps_history.append(1000.0/elapsed if elapsed > 0 else 0)

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def loop(self):
        while True:
            self.process_counter += 1
            if self.process_counter >= CFG.AI_FRAME_SKIP and self.latest_frame is not None:
                self.process_counter = 0
                with self.lock: work_frame = self.latest_frame.copy()
                self.process(work_frame)
            time.sleep(0.01)

# ================= 🖥️ UI & MAIN =================
def draw_ui(frame, ai_proc):
    with ai_proc.lock:
        res, ms = ai_proc.results.copy(), ai_proc.ms
        fps = np.mean(ai_proc.fps_history) if ai_proc.fps_history else 0

    # Draw Boxes
    for r in res:
        x1, y1, x2, y2 = r['box']
        c = (0, 255, 0) if r['conf'] > CFG.VERIFY_THRESHOLD else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        draw_text(frame, f"{r['label']} {r['conf']:.2f}", (x1, y1-10), 0.4, c)

    # Draw Checklist
    y_p = 40
    for d in ai_proc.rx.all_drugs:
        v = d in ai_proc.rx.verified
        col = (0, 255, 0) if v else (180, 180, 180)
        draw_text(frame, f"{'√' if v else 'x'} {d.upper()}", (CFG.DISPLAY_SIZE[0]-150, y_p), 0.5, col)
        y_p += 25

    draw_text(frame, f"AI: {ms:.1f}ms | FPS: {fps:.1f}", (10, 20), 0.5, (0, 255, 255))

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, CFG.DISPLAY_SIZE[0]); cap.set(4, CFG.DISPLAY_SIZE[1])
    ai = AIProcessor().start()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        with ai.lock: ai.latest_frame = frame
        
        display = frame.copy()
        draw_ui(display, ai)
        cv2.imshow("PillTrack Optimized", display)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()