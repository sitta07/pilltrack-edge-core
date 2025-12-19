#!/usr/bin/env python3
"""
PILLTRACK – HYBRID ENGINE (DINO 85% + SIFT 15%)
✔ Detailed Console Logs (Human Readable)
✔ RGB8888 END-TO-END
✔ Checklist UI
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

from dataclasses import dataclass
from typing import Tuple, List, Dict
from torchvision import transforms
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
    
    # Weight Settings (85/15)
    W_DINO: float = 0.6
    W_SIFT: float = 0.4
    SIFT_SATURATION: int = 400

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
        if not os.path.exists(CFG.DRUG_LIST_JSON):
            return
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
                    print(f"🎉 VERIFIED: {original.upper()} added to checklist!")
                    self.verified.add(original)
                return True
        return False

# ================= 🎨 FEATURE ENGINE (HYBRID) =================
class FeatureEngine:
    def __init__(self):
        # ใช้ vitb14 (ตัวใหญ่) ตาม Database เพื่อแก้ Error shape mismatch
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval().to(device)
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
        
        self.sift = cv2.SIFT_create()

    @torch.no_grad()
    def extract(self, img):
        # DINO
        v = self.model(self.tf(img).unsqueeze(0).to(device))
        v = v.flatten().cpu().numpy()
        dino_vec = v / (np.linalg.norm(v)+1e-8)
        
        # SIFT
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, sift_des = self.sift.detectAndCompute(gray, None)
        
        return dino_vec, sift_des

# ================= 🤖 AI PROCESSOR =================
class AIProcessor:
    def __init__(self):
        self.rx = PrescriptionManager()
        self.engine = FeatureEngine()
        self.db_dino = {} 
        self.db_sift = {}
        self.bf = cv2.BFMatcher()
        self.load_db()
        self.yolo = YOLO(CFG.MODEL_PACK)
        self.latest = None
        self.lock = threading.Lock()
        self.ms = 0
        
        # ตัวแปรสำหรับคุม Log ไม่ให้พิมพ์รัวเกินไป
        self.last_log_time = 0 

    def load_db(self):
        if not os.path.exists(CFG.DB_PACKS_VEC):
            print(f"❌ Error: Database {CFG.DB_PACKS_VEC} not found")
            return

        with open(CFG.DB_PACKS_VEC,'rb') as f:
            raw = pickle.load(f)
            
        print(f"📦 Loading Database: {len(raw)} items")
        for name, data in raw.items():
            if isinstance(data, dict):
                self.db_dino[name] = [np.array(v) for v in data.get('dino', [])]
                self.db_sift[name] = data.get('sift', [])
            elif isinstance(data, list):
                self.db_dino[name] = [np.array(v) for v in data]
                self.db_sift[name] = []

    def get_sift_score(self, query_des, target_des_list):
        if query_des is None or not target_des_list: return 0.0
        max_score = 0.0
        for target_des in target_des_list:
            if target_des is None: continue
            try:
                matches = self.bf.knnMatch(query_des, target_des, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                score = min(len(good) / CFG.SIFT_SATURATION, 1.0)
                if score > max_score: max_score = score
            except: continue
        return max_score

    def process(self, frame):
        t0 = time.time()
        img = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        res = self.yolo(img, conf=CFG.CONF_THRESHOLD, imgsz=CFG.AI_SIZE, verbose=False, task='segment')[0]

        if res.masks is None: return

        sx = CFG.DISPLAY_SIZE[0]/CFG.AI_SIZE
        sy = CFG.DISPLAY_SIZE[1]/CFG.AI_SIZE

        for box, mask in zip(res.boxes, res.masks):
            x1,y1,x2,y2 = box.xyxy[0].int().tolist()
            rx1,ry1,rx2,ry2 = int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
            
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0: continue

            q_vec, q_des = self.engine.extract(crop)

            # --- STEP 1: DINO Filter ---
            candidates = []
            for name, db_vecs in self.db_dino.items():
                best_sim = -1.0
                for v in db_vecs:
                    sim = np.dot(q_vec, v)
                    if sim > best_sim: best_sim = sim
                if best_sim > 0.4:
                    candidates.append((name, best_sim))

            candidates.sort(key=lambda x: x[1], reverse=True)
            top_k = candidates[:5]

            # --- STEP 2: SIFT Refine ---
            best_final_score = -1
            best_label = "Unknown"
            
            # เก็บค่าแยกไว้แสดง Log
            log_dino_score = 0.0
            log_sift_score = 0.0

            for name, dino_score in top_k:
                sift_score = self.get_sift_score(q_des, self.db_sift.get(name, []))
                final_score = (dino_score * CFG.W_DINO) + (sift_score * CFG.W_SIFT)

                if final_score > best_final_score:
                    best_final_score = final_score
                    best_label = name
                    log_dino_score = dino_score
                    log_sift_score = sift_score

            # --- LOGGING SECTION (พิมพ์ Log สวยๆ) ---
            # พิมพ์ทุกๆ 2 วินาที หรือถ้าเจอยาที่มั่นใจมากๆ
            current_time = time.time()
            if (current_time - self.last_log_time > 1.0) and (best_final_score > 0.5):
                self.last_log_time = current_time
                
                # แปลงเป็น % ให้ดูง่าย
                total_pct = best_final_score * 100
                dino_pct = log_dino_score * 100
                sift_pct = log_sift_score * 100
                
                # แยกสี Log (เขียว=ผ่าน, เหลือง=เกือบ, แดง=ไม่ผ่าน)
                status_icon = "✅" if best_final_score > 0.60 else "⚠️"
                
                print("-" * 50)
                print(f"{status_icon} ผลการตรวจจับ: {best_label.upper()}")
                print(f"   🏆 ความมั่นใจรวม: {total_pct:.1f}%")
                print(f"   🔹 ความเหมือนรูปทรง (DINO): {dino_pct:.1f}%  (น้ำหนัก {CFG.W_DINO})")
                print(f"   🔸 รายละเอียดจุดภาพ (SIFT): {sift_pct:.1f}%  (น้ำหนัก {CFG.W_SIFT})")
                print("-" * 50)

            # Decision
            if best_final_score > 0.60: 
                self.rx.verify(best_label)

        self.ms = (time.time()-t0)*1000

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def loop(self):
        while True:
            if self.latest is not None:
                self.process(self.latest)
            time.sleep(0.01)

# ================= 📷 CAMERA =================
class Camera:
    def __init__(self):
        try:
            from picamera2 import Picamera2
            self.cam = Picamera2()
            self.cam.configure(
                self.cam.create_preview_configuration(
                    main={"size":CFG.DISPLAY_SIZE,"format":"RGB888"}
                )
            )
            self.cam.start()
            self.pi = True
            print("📷 PiCamera2 initialized.")
        except:
            self.cap = cv2.VideoCapture(0)
            self.pi = False
            print("📷 USB Webcam initialized.")

    def get(self):
        if self.pi:
            return self.cam.capture_array()
        ret,f = self.cap.read()
        return f if ret else None

# ================= 🖥️ UI =================
def draw_ui(frame, rx):
    h, w = frame.shape[:2]
    right_margin = w - 10
    y = 40

    for drug in rx.all_drugs:
        done = drug in rx.verified
        color = (0,255,0) if done else (180,180,180)
        text = f"✔ {drug.upper()}" if done else f"□ {drug.upper()}"

        (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 2)
        x = right_margin - tw

        draw_text(frame, text, (x, y), 0.55, color)

        if done:
            cv2.line(frame, (x, y-8), (x+tw, y-8), (0,255,0), 2)
        y += 28

# ================= 🚀 MAIN =================
if __name__ == "__main__":
    try:
        SyncManager().sync()
    except Exception as e:
        print(f"⚠️ Sync Error: {e}")

    cam = Camera()
    ai = AIProcessor().start()

    cv2.namedWindow("PillTrack", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PillTrack", *CFG.DISPLAY_SIZE)

    print("🚀 System Started (Hybrid Mode). Press 'q' to exit.")

    while True:
        frame = cam.get()
        if frame is None:
            continue

        ai.latest = frame.copy()
        draw_ui(frame, ai.rx)
        cv2.putText(frame, f"AI: {ai.ms:.1f}ms", (10, 20), FONT, 0.5, (0,255,255), 1)
        cv2.imshow("PillTrack", frame)

        if cv2.waitKey(1) == ord('q'):
            break
            
    cv2.destroyAllWindows()