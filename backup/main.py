#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║ PILLTRACK: GOD MODE (Numpy Matrix Edition)                   ║
║ - Logic: Pure Math (Dot Product) -> เหมือนโค้ดเก่าที่แม่นๆ     ║
║ - Engine: DINOv2 (448px) + YOLOv8                            ║
║ - No Qdrant: ตัดทิ้งเลย ใช้ Numpy คำนวณสด (เร็วกว่าสำหรับ Pi)   ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import cv2
import torch
import pickle
from torchvision import transforms
from ultralytics import YOLO

# ================= ⚙️ CONFIGURATION =================
@dataclass
class Config:
    # --- PATHS (แก้ให้ตรงเป๊ะๆ) ---
    MODEL_PACK: str = 'models/seg_best_process.pt'
    
    # 📂 DATABASE FILES
    DB_PILLS_VEC: str = 'database/db_register/db_pills_dino.pkl'
    DB_PACKS_VEC: str = 'database/db_register/db_packs_dino.pkl'
    
    DB_PILLS_COL: str = 'database/db_register/colors_pills.pkl'
    DB_PACKS_COL: str = 'database/db_register/colors_packs.pkl'
    
    IMG_DB_FOLDER: str = 'database_images' 
    PRESCRIPTION_FILE: str = 'prescription.txt'

    # Display
    DISPLAY_SIZE: Tuple[int, int] = (640, 480) # Pi แนะนำไซส์นี้
    AI_SIZE: int = 320 # ลดภาระ YOLO
    
    # 🔥 DINO CONFIG (ต้องตรงกับ build_db)
    DINO_SIZE: int = 448 
    VEC_DIM: int = 384
    
    # Exclusion Zone
    UI_ZONE_X_START: int = 400
    UI_ZONE_Y_END: int = 150
    
    # 🎚️ TUNING (จูนความแม่นตรงนี้)
    CONF_THRESHOLD: float = 0.6 # ถ้ายัง Unknown ให้ลดเลขนี้ลง (เช่น 0.25)
    
    # Weights (สูตรลับความแม่น)
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {'vec': 0.4, 'col': 0.2, 'sift': 0.4})
    SIFT_RATIO_TEST: float = 0.75

CFG = Config()
device = torch.device("cpu") # Pi ใช้ CPU
print(f"🚀 SYSTEM STARTING: (Numpy Matrix)")

# ================= 🧠 PRESCRIPTION MANAGER =================
class PrescriptionManager:
    def __init__(self):
        self.patient_name = "Unknown"
        self.allowed_drugs = [] # ถ้าลิสต์นี้ว่าง คือยอมให้ผ่านทุกยา (Debug Mode)
        self.verified_drugs = set()
        self.load_prescription()

    def load_prescription(self):
        if not os.path.exists(CFG.PRESCRIPTION_FILE): return
        try:
            with open(CFG.PRESCRIPTION_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line or line.startswith('#'): continue
                    parts = line.split('|')
                    if len(parts) >= 3:
                        self.patient_name = parts[1].strip()
                        raw_drugs = parts[2].split(',')
                        self.allowed_drugs = [d.strip().lower() for d in raw_drugs if d.strip()]
                        print(f"📋 Rx Target: {self.allowed_drugs}")
                        break
        except: pass

    def is_allowed(self, name):
        # ถ้าไม่มีใบยา (List ว่าง) ให้ถือว่า "อนุญาตหมด" (จะได้ไม่ขึ้น Unknown มั่วๆ)
        if not self.allowed_drugs: return True
        
        clean = name.lower().split('_rot')[0].replace('_pack','').replace('_pill','')
        for allowed in self.allowed_drugs:
            if allowed in clean or clean in allowed: return True
        return False

    def verify(self, name):
        clean = name.lower().split('_rot')[0].replace('_pack','').replace('_pill','')
        self.verified_drugs.add(clean)

# ================= 🎨 FEATURE ENGINE =================
class FeatureEngine:
    def __init__(self):
        print("🦕 Loading DINOv2 (ViT-S/14)...")
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model.eval().to(device)
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((CFG.DINO_SIZE, CFG.DINO_SIZE), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("✅ DINOv2 Ready")
        except Exception as e:
            print(f"❌ DINO Error: {e}"); sys.exit(1)

        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    @torch.no_grad()
    def get_vector(self, img_rgb):
        t = self.preprocess(img_rgb).unsqueeze(0).to(device)
        output = self.model(t)
        if isinstance(output, dict): output = output['x_norm_clstoken']
        vec = output.flatten().cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-8)

    def get_sift_features(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return self.sift.detectAndCompute(gray, None)[1]

# ================= 🤖 AI PROCESSOR (NUMPY EDITION) =================
class AIProcessor:
    def __init__(self):
        self.engine = FeatureEngine()
        self.rx_manager = PrescriptionManager()
        
        # 🔥 DATA STRUCTURES (RAM)
        self.db_keys = []      # เก็บชื่อยา ["para_rot0", "para_rot90", ...]
        self.db_vectors = []   # เก็บ Vector [N, 384]
        self.db_colors = {}
        self.db_sift = {}
        
        self._load_database()
        
        try:
            print(f"📦 Loading YOLO...")
            self.yolo_pack = YOLO(CFG.MODEL_PACK)
        except: 
            self.yolo_pack = YOLO('yolov8n-seg.pt')

        self.latest_frame = None
        self.results = []
        self.lock = threading.Lock()
        self.stopped = False

    def _load_database(self):
        print("🔍 Loading Database into RAM (Numpy Matrix)...")
        
        # 1. Load Vectors
        temp_vecs = []
        
        def load_vec_file(path):
            if os.path.exists(path):
                with open(path, 'rb') as f: return pickle.load(f)
            return {}

        raw_data = {**load_vec_file(CFG.DB_PILLS_VEC), **load_vec_file(CFG.DB_PACKS_VEC)}
        
        for key, val in raw_data.items():
            # Check Permission (Rx)
            # ถ้า Rx ว่าง = โหลดหมด, ถ้า Rx มีของ = โหลดเฉพาะที่ตรง
            if not self.rx_manager.is_allowed(key): continue
            
            # Extract Vector
            vec = val[0] if isinstance(val, list) and len(val) > 0 else val
            if isinstance(vec, list): vec = vec[0]
            if len(vec) != CFG.VEC_DIM: continue
            
            self.db_keys.append(key)
            temp_vecs.append(vec)

        if temp_vecs:
            self.db_vectors = np.array(temp_vecs) # แปลงเป็น Matrix ก้อนใหญ่
            print(f"✅ Vectors Loaded: {len(self.db_keys)} items")
        else:
            print("⚠️ Warning: No vectors loaded! (Check Paths or Rx Filter)")

        # 2. Load Colors
        def load_col_file(path):
            if os.path.exists(path):
                with open(path, 'rb') as f: self.db_colors.update(pickle.load(f))
        load_col_file(CFG.DB_PILLS_COL)
        load_col_file(CFG.DB_PACKS_COL)

        # 3. Load SIFT
        if os.path.exists(CFG.IMG_DB_FOLDER):
            for d in os.listdir(CFG.IMG_DB_FOLDER):
                if not self.rx_manager.is_allowed(d): continue
                p = os.path.join(CFG.IMG_DB_FOLDER, d)
                if os.path.isdir(p):
                    des = []
                    for f in sorted(os.listdir(p))[:2]: # Load 2 images max
                        if f.endswith(('jpg','png')):
                            im = cv2.imread(os.path.join(p, f))
                            if im is not None:
                                des.append(self.engine.get_sift_features(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
                    if des: self.db_sift[d] = des

    def compute_sift_score(self, query_des, target_key):
        base_name = target_key.split('_rot')[0].replace('_pack','').replace('_pill','')
        # พยายาม Match ชื่อยาให้ตรงกับ Folder SIFT
        target_folder = None
        for k in self.db_sift.keys():
            if base_name in k or k in base_name:
                target_folder = k
                break
        
        if query_des is None or target_folder is None: return 0.0
        
        max_m = 0
        for ref_des in self.db_sift[target_folder]:
            try:
                matches = self.engine.bf.knnMatch(query_des, ref_des, k=2)
                good = len([m for m,n in matches if m.distance < CFG.SIFT_RATIO_TEST*n.distance])
                max_m = max(max_m, good)
            except: pass
        return min(max_m / 15.0, 1.0) # 15 จุดถือว่าเต็ม

    def match(self, vec, img_crop):
        if len(self.db_vectors) == 0: return []

        # 🔥 MATRIX MULTIPLICATION (เร็วและแม่นเหมือน np.dot)
        # scores = [Cosine Similarity ของทุกตัวใน DB]
        scores = np.dot(self.db_vectors, vec)
        
        # เอา Top 5
        top_indices = np.argsort(scores)[::-1][:5]
        
        candidates = []
        query_sift = self.engine.get_sift_features(img_crop)

        for idx in top_indices:
            key = self.db_keys[idx]
            vec_score = scores[idx]
            
            # Color Score
            col_score = 0.5
            base = key.split('_rot')[0]
            if base in self.db_colors: col_score = 0.8 # Placeholder Check
            
            # SIFT Score
            sift_score = self.compute_sift_score(query_sift, key)
            
            # Final Fusion
            final = (vec_score * CFG.WEIGHTS['vec']) + \
                    (col_score * CFG.WEIGHTS['col']) + \
                    (sift_score * CFG.WEIGHTS['sift'])
            
            candidates.append((key, final, vec_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Debug Print (เอาไว้ดูว่าทำไม Unknown)
        if candidates:
            print(f"🔎 Best Match: {candidates[0][0]} | Score: {candidates[0][1]:.2f} (Vec:{candidates[0][2]:.2f})")
            
        return candidates

    def process_frame(self, frame):
        img_ai = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        res = self.yolo_pack(img_ai, verbose=False, conf=0.5, imgsz=CFG.AI_SIZE)
        
        detections = []
        if len(res[0].boxes) == 0:
            with self.lock: self.results = []; return

        sx, sy = CFG.DISPLAY_SIZE[0]/CFG.AI_SIZE, CFG.DISPLAY_SIZE[1]/CFG.AI_SIZE
        
        for box in res[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            rx1, ry1, rx2, ry2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
            
            cx, cy = (rx1+rx2)//2, (ry1+ry2)//2
            if cx > CFG.UI_ZONE_X_START and cy < CFG.UI_ZONE_Y_END: continue

            h, w = frame.shape[:2]
            pad = 5
            cx1, cy1 = max(0, rx1-pad), max(0, ry1-pad)
            cx2, cy2 = min(w, rx2+pad), min(h, ry2+pad)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0: continue

            vec = self.engine.get_vector(crop)
            candidates = self.match(vec, crop)
            
            label = "Unknown"
            score = 0.0
            if candidates:
                top_key, val, _ = candidates[0]
                if val > CFG.CONF_THRESHOLD:
                    label = top_key
                    self.rx_manager.verify(label)
                score = val

            detections.append({'box':(rx1,ry1,rx2,ry2), 'label':label, 'score':score})
            
        with self.lock: self.results = detections

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self
    def _run(self):
        while not self.stopped:
            with self.lock: f = self.latest_frame
            if f is not None:
                try: self.process_frame(f)
                except Exception as e: print(f"Err: {e}")
            time.sleep(0.01)

# ================= 📷 CAMERA =================
class Camera:
    def __init__(self):
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            c = self.picam.create_preview_configuration(main={"size": CFG.DISPLAY_SIZE, "format": "RGB888"})
            self.picam.configure(c); self.picam.start(); self.use_pi = True
            print("📷 PiCamera2 Active")
        except:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, CFG.DISPLAY_SIZE[0]); self.cap.set(4, CFG.DISPLAY_SIZE[1])
            self.use_pi = False
            print("📷 USB Camera Active")
    def get(self):
        return self.picam.capture_array() if self.use_pi else cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)
    def stop(self):
        self.picam.stop() if self.use_pi else self.cap.release()

# ================= 🚀 MAIN =================
if __name__ == "__main__":
    cam = Camera()
    ai = AIProcessor().start()
    
    cv2.namedWindow("PillTrack", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            frame = cam.get()
            if frame is None: continue
            
            ai.latest_frame = frame.copy()
            
  
            # Dashboard
            cv2.rectangle(frame, (CFG.UI_ZONE_X_START, 0), (640, CFG.UI_ZONE_Y_END), (0,0,0), -1)
            y = 20
            # Show Rx Status
            cv2.putText(frame, f"Rx: {ai.rx_manager.patient_name}", (CFG.UI_ZONE_X_START+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y+=20
            
            for drug in ai.rx_manager.allowed_drugs:
                verified = any(drug in v for v in ai.rx_manager.verified_drugs)
                c = (0,255,0) if verified else (100,100,100)
                icon = "[/]" if verified else "[ ]"
                cv2.putText(frame, f"{icon} {drug}", (CFG.UI_ZONE_X_START+10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
                y += 20
            
            cv2.imshow("PillTrack GodMode", frame)
            if cv2.waitKey(1) == ord('q'): break
    except KeyboardInterrupt: pass
    finally: cam.stop(); ai.stopped=True; cv2.destroyAllWindows()