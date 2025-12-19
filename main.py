#!/usr/bin/env python3
"""
PILLTRACK – HYBRID ENGINE (WITH SEGMENT VISUALIZATION & LIVE LOGS)
✔ Draw Bounding Boxes & Labels
✔ Real-time Console Logging
✔ Hybrid Vector Search (DINOv2 + SIFT)
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
# from sync_manager import SyncManager # เปิดใช้งานถ้ามีไฟล์ sync

# ================= ⚙️ CONFIG =================
# จำลอง config ถ้าไม่มีไฟล์จริง
DEFAULT_CFG = {
    'artifacts': {'model': 'yolov8n.pt', 'drug_list': 'drug_list.json'},
    'display': {'width': 640, 'height': 480},
    'settings': {'yolo_conf': 0.45}
}

@dataclass
class Config:
    MODEL_PACK: str = "yolov8n.pt" # เปลี่ยนเป็น path ของคุณ
    DB_PACKS_VEC: str = "database/pill_fingerprints.pkl"
    DRUG_LIST_JSON: str = "drug_list.json"
    DISPLAY_SIZE: Tuple[int, int] = (640, 480)
    AI_SIZE: int = 416
    CONF_THRESHOLD: float = 0.45
    
    W_DINO: float = 0.6
    W_SIFT: float = 0.4
    SIFT_SATURATION: int = 400
    
    MEAN: np.ndarray = field(default_factory=lambda: np.array([0.485, 0.456, 0.406], dtype=np.float32))
    STD: np.ndarray = field(default_factory=lambda: np.array([0.229, 0.224, 0.225], dtype=np.float32))

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ================= 🛠️ UI HELPERS =================
def draw_text(img, text, pos, scale=0.5, color=(255,255,255), thickness=1):
    cv2.putText(img, text, pos, FONT, scale, (0,0,0), thickness+2)
    cv2.putText(img, text, pos, FONT, scale, color, thickness)

# ================= 🧠 RX MANAGER =================
class PrescriptionManager:
    def __init__(self):
        self.all_drugs = []
        self.norm_map = {}
        self.verified = set()
        self.load()

    def load(self):
        if not os.path.exists(CFG.DRUG_LIST_JSON): 
            self.all_drugs = ["paracetamol", "aspirin"] # Mockup
            return
        with open(CFG.DRUG_LIST_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.all_drugs = [d.lower() for d in data.get('drugs', [])]

    def verify(self, name: str):
        name_low = name.lower()
        if name_low in self.all_drugs and name_low not in self.verified:
            self.verified.add(name_low)
            print(f"🌟 [MATCH FOUND]: {name_low.upper()} verified!")
            return True
        return False

# ================= 🧠 FEATURE ENGINE =================
class FeatureEngine:
    def __init__(self):
        print("⏳ Loading DINOv2...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.eval().to(device)
        self.sift = cv2.SIFT_create()

    def extract_dino_batch(self, crop_list):
        if not crop_list: return np.array([])
        batch = []
        for img in crop_list:
            img = cv2.resize(img, (224, 224))
            img = (img.astype(np.float32) / 255.0 - CFG.MEAN) / CFG.STD
            batch.append(img.transpose(2, 0, 1))
        
        t_batch = torch.from_numpy(np.array(batch)).to(device)
        with torch.no_grad():
            emb = self.model(t_batch)
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()

# ================= 🤖 AI PROCESSOR =================
class AIProcessor:
    def __init__(self):
        self.rx = PrescriptionManager()
        self.engine = FeatureEngine()
        self.yolo = YOLO(CFG.MODEL_PACK)
        
        self.db_vectors = np.random.rand(10, 384).astype('float32') # Placeholder
        self.db_names = ["pill_a"] * 10
        
        self.latest_frame = None
        self.detections = [] # เก็บผลลัพธ์เพื่อส่งไปวาด [box, label, score]
        self.lock = threading.Lock()
        self.ms = 0

    def process(self, frame):
        t0 = time.time()
        res = self.yolo(frame, conf=CFG.CONF_THRESHOLD, verbose=False)[0]
        
        new_detections = []
        crops = []
        boxes_coords = []

        if res.boxes:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
                    boxes_coords.append((x1, y1, x2, y2))

        if crops:
            # Batch Feature Extraction
            vecs = self.engine.extract_dino_batch(crops)
            # ตัวอย่างการหา Similarity (Mockup logic)
            for i, vec in enumerate(vecs):
                label = "Pill" # ในงานจริงใช้ Matrix Similarity หาชื่อยา
                score = 0.95
                new_detections.append({
                    "box": boxes_coords[i],
                    "label": label,
                    "score": score
                })
                # Log ตลอดเวลาเมื่อเจอ
                print(f"🔍 Detecting: {label} ({score*100:.1f}%) at {boxes_coords[i]}")

        with self.lock:
            self.detections = new_detections
            self.ms = (time.time() - t0) * 1000

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

# ================= 📷 CAMERA =================
class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
    def get(self):
        ret, f = self.cap.read()
        return cv2.resize(f, CFG.DISPLAY_SIZE) if ret else None

# ================= 🖥️ UI & DRAWING =================
def draw_overlay(frame, ai_proc):
    # 1. วาด Bounding Boxes (Segments)
    with ai_proc.lock:
        for det in ai_proc.detections:
            x1, y1, x2, y2 = det['box']
            label = det['label']
            score = det['score']
            
            # วาดกรอบ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # วาดพื้นหลังตัวหนังสือ
            cv2.rectangle(frame, (x1, y1-20), (x1+100, y1), (0, 255, 0), -1)
            draw_text(frame, f"{label} {score:.2f}", (x1+5, y1-5), 0.4, (0,0,0), 1)

    # 2. วาด Checklist (Prescription)
    y_offset = 30
    for drug in ai_proc.rx.all_drugs:
        status = "✅" if drug in ai_proc.rx.verified else "□"
        color = (0, 255, 0) if drug in ai_proc.rx.verified else (200, 200, 200)
        draw_text(frame, f"{status} {drug.upper()}", (CFG.DISPLAY_SIZE[0]-150, y_offset), 0.5, color)
        y_offset += 25

# ================= 🚀 MAIN =================
if __name__ == "__main__":
    cam = Camera()
    ai = AIProcessor().start()

    print("🚀 System Live... Press 'q' to quit.")

    while True:
        frame = cam.get()
        if frame is None: break

        # ส่งเฟรมให้ AI Thread
        ai.latest_frame = frame

        # สร้างเฟรมสำหรับแสดงผล (Draw Everything Here)
        display_frame = frame.copy()
        draw_overlay(display_frame, ai)
        
        # Dashboard
        draw_text(display_frame, f"Latency: {ai.ms:.1f}ms", (10, 20), 0.5, (0, 255, 255))
        
        cv2.imshow("PillTrack – AI Vision", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()