#!/usr/bin/env python3
"""
PILLTRACK – SENIOR EDITION (AUDIT MODE / UI INTERACTIVE)
✔ Full Search: Detects both Prescribed AND Unexpected items
✔ Split UI: Shows "Prescription Checklist" vs "⚠️ Extra Items"
✔ Mouse Interaction: Click to toggle manual check (Mockup Logic)
✔ RGB8888 Pipeline
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
from typing import Tuple, List, Dict, Optional
from ultralytics import YOLO
from collections import deque
from his_connector import HISConnector

try:
    from sync_manager import SyncManager
except ImportError:
    SyncManager = None

# ================= ⚙️ CONFIG =================
with open("config.yaml", "r") as f:
    yaml_cfg = yaml.safe_load(f)

@dataclass
class Config:
    MODEL_PACK: str = os.path.abspath(yaml_cfg['artifacts']['model'])
    DB_PACKS_VEC: str = "database/pill_fingerprints.pkl"
    DRUG_LIST_JSON: str = yaml_cfg['artifacts']['drug_list']
    DISPLAY_SIZE: Tuple[int, int] = (
        yaml_cfg['display']['width'],
        yaml_cfg['display']['height']
    )
    AI_SIZE: int = 416
    CONF_THRESHOLD: float = yaml_cfg['settings']['yolo_conf']
    MODE: str = yaml_cfg.get('operation', {}).get('mode', 'standalone')
    
    # Scoring
    W_DINO: float = 0.6
    W_SIFT: float = 0.4
    SIFT_SATURATION: int = 400
    SIFT_TOP_K: int = 3
    DINO_TOP_K: int = 5
    
    # Performance
    AI_FRAME_SKIP: int = 2
    MIN_DINO_SCORE: float = 0.4
    VERIFY_THRESHOLD: float = 0.60 
    
    MEAN: np.ndarray = field(default_factory=lambda: np.array([0.485, 0.456, 0.406], dtype=np.float32))
    STD: np.ndarray = field(default_factory=lambda: np.array([0.229, 0.224, 0.225], dtype=np.float32))

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ================= 📷 CAMERA HANDLER =================
class CameraHandler:
    def __init__(self, width=1280, height=720):
        self.width, self.height = width, height
        self.use_picamera = False
        self.cap, self.picam = None, None
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            config = self.picam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "XRGB8888"}
            )
            self.picam.configure(config)
            self.picam.start()
            self.use_picamera = True
            print("📷 Camera: Using Picamera2 (XRGB8888)")
        except:
            self.use_picamera = False
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, self.width); self.cap.set(4, self.height)

    def get_frame(self):
        if self.use_picamera: return self.picam.capture_array()
        else:
            ret, frame = self.cap.read()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) if ret else None

    def release(self):
        if self.use_picamera: self.picam.stop()
        elif self.cap: self.cap.release()

# ================= 🧠 PRESCRIPTION MANAGER =================
class PrescriptionManager:
    def __init__(self):
        self.target_drugs = {} 
        self.detected_extras = set() # เก็บรายชื่อยาที่ "ไม่อยู่ในใบสั่ง"
        self.patient_name = "Standalone"
        self.is_ready = False
        
        if CFG.MODE == "standalone":
            self.load_local_all()
            self.is_ready = True

    def load_local_all(self):
        if not os.path.exists(CFG.DRUG_LIST_JSON): return
        with open(CFG.DRUG_LIST_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for d in data.get('drugs', []):
            self.target_drugs[normalize_name(d)] = {"original": d, "qty": None, "found": 0, "manual": False}

    def update_from_his(self, his_data: Dict):
        self.target_drugs = {}
        self.detected_extras = set()
        self.patient_name = his_data.get('patient_name', 'Unknown')
        for item in his_data.get('prescription', []):
            norm = normalize_name(item['name'])
            self.target_drugs[norm] = {
                "original": item['name'],
                "qty": item['amount'],
                "found": 0,
                "manual": False # เผื่อติ๊กมือ
            }
        self.is_ready = True

    def verify(self, detected_name: str):
        norm_det = normalize_name(detected_name)
        if norm_det in self.target_drugs:
            self.target_drugs[norm_det]['found'] = 1 # Auto-tick
            return True # Is Valid
        else:
            self.detected_extras.add(detected_name)
            return False # Is Extra

    def reset(self):
        self.target_drugs = {}
        self.detected_extras = set()
        self.patient_name = "Waiting..."
        self.is_ready = False

# ================= 🛠️ UTILS =================
def normalize_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', re.sub(r'_pack.*', '', name.lower()))

def draw_text(img, text, pos, scale=0.5, color=(255,255,255,255), thickness=1):
    cv2.putText(img, text, pos, FONT, scale, (0,0,0,255), thickness+2)
    cv2.putText(img, text, pos, FONT, scale, color, thickness)

# ================= 🔍 FEATURE ENGINE =================
class FeatureEngine:
    def __init__(self):
        print("⏳ Loading DINOv2...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.eval().to(device)
        if device.type == 'cuda': self.model = self.model.half()
        self.sift = cv2.SIFT_create(nfeatures=500)
        
    def preprocess_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        batch = np.zeros((len(crop_list), 3, 224, 224), dtype=np.float32)
        for i, img in enumerate(crop_list):
            img_rgb = img[:, :, :3] 
            img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            img_norm = (img_resized.astype(np.float32) / 255.0 - CFG.MEAN) / CFG.STD
            batch[i] = img_norm.transpose(2, 0, 1)
        return batch

    @torch.no_grad()
    def extract_dino_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        if not crop_list: return np.array([])
        img_batch_np = self.preprocess_batch(crop_list)
        img_batch_t = torch.from_numpy(img_batch_np).to(device)
        if device.type == 'cuda': img_batch_t = img_batch_t.half()
        embeddings = self.model(img_batch_t)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().float().numpy()

    def extract_sift(self, img: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        _, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors

# ================= 🤖 AI PROCESSOR =================
class AIProcessor:
    def __init__(self):
        self.rx = PrescriptionManager()
        self.engine = FeatureEngine()
        self.his = HISConnector()
        
        # Load FULL DB (To detect extras)
        self.full_db_vectors, self.full_db_sift = {}, {}
        self.full_vectors_matrix = []
        self.full_names_list = []
        
        self.bf = cv2.BFMatcher()
        self.load_db()
        
        print("⏳ Loading YOLO...")
        self.yolo = YOLO(CFG.MODEL_PACK)
        
        self.latest_frame = None
        self.results = []
        self.lock = threading.Lock()
        self.process_counter = 0

    def load_db(self):
        if not os.path.exists(CFG.DB_PACKS_VEC): return
        with open(CFG.DB_PACKS_VEC, 'rb') as f: raw = pickle.load(f)
        
        vectors = []
        for name, data in raw.items():
            norm = normalize_name(name)
            dino_list = data.get('dino', []) if isinstance(data, dict) else data
            sift_list = data.get('sift', []) if isinstance(data, dict) else []
            self.full_db_vectors[norm] = dino_list
            self.full_db_sift[norm] = sift_list[:CFG.SIFT_TOP_K]
            
            for vec in dino_list:
                vectors.append(vec)
                self.full_names_list.append(norm)
                
        self.full_vectors_matrix = np.array(vectors, dtype=np.float32).T
        print(f"✅ Database loaded: {len(self.full_db_vectors)} drugs.")

    def get_sift_score(self, query_des, target_des_list) -> float:
        if query_des is None or not target_des_list: return 0.0
        max_score = 0.0
        for target_des in target_des_list:
            if target_des is None or len(target_des) < 2: continue
            matches = self.bf.knnMatch(query_des, target_des, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            max_score = max(max_score, min(len(good) / CFG.SIFT_SATURATION, 1.0))
        return max_score

    def process(self, frame: np.ndarray):
        if not self.rx.is_ready: return
        t1 = time.perf_counter()
        
        img_rgb = frame[:, :, :3]
        img_resized = cv2.resize(img_rgb, (CFG.AI_SIZE, CFG.AI_SIZE))
        res = self.yolo(img_resized, conf=CFG.CONF_THRESHOLD, verbose=False)[0]

        if res.boxes is None or len(res.boxes) == 0:
            with self.lock: self.results = []
            return

        sx, sy = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE, CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
        crops, box_coords = [], []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            dx1, dy1, dx2, dy2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
            crop = frame[max(0, dy1):min(frame.shape[0], dy2), max(0, dx1):min(frame.shape[1], dx2)]
            if crop.size > 0: crops.append(crop); box_coords.append([dx1, dy1, dx2, dy2])

        temp_results = []
        if crops:
            batch_dino = self.engine.extract_dino_batch(crops)
            # Full Search (To find extras)
            sim_matrix = np.dot(batch_dino, self.full_vectors_matrix)
            
            for i, crop in enumerate(crops):
                best_idx = np.argmax(sim_matrix[i])
                dino_score = sim_matrix[i][best_idx]
                if dino_score < CFG.MIN_DINO_SCORE: continue
                
                matched_norm = self.full_names_list[best_idx]
                
                q_des = self.engine.extract_sift(crop)
                sift_score = self.get_sift_score(q_des, self.full_db_sift.get(matched_norm, []))
                fusion = (dino_score * CFG.W_DINO) + (sift_score * CFG.W_SIFT)
                
                # Check status: VALID or EXTRA
                is_valid = self.rx.verify(matched_norm) if fusion > CFG.VERIFY_THRESHOLD else False
                
                temp_results.append({
                    'box': box_coords[i], 
                    'label': matched_norm.upper(), 
                    'conf': fusion,
                    'is_valid': is_valid
                })

        with self.lock: self.results = temp_results
        # print(f"Processing: {(time.perf_counter()-t1)*1000:.1f}ms")

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def loop(self):
        while True:
            self.process_counter += 1
            if self.process_counter >= CFG.AI_FRAME_SKIP and self.latest_frame is not None:
                self.process_counter = 0
                self.process(self.latest_frame.copy())
            time.sleep(0.01)

# ================= 🖥️ UI & DISPLAY =================
def draw_ui(frame: np.ndarray, ai_proc: AIProcessor):
    rx = ai_proc.rx
    
    # Colors (RGBA)
    C_GREEN = (0, 255, 0, 255)
    C_RED = (255, 0, 0, 255)   # For Warning
    C_GRAY = (180, 180, 180, 255)
    C_WHITE = (255, 255, 255, 255)
    C_YELLOW = (255, 255, 0, 255)

    # 1. Header
    draw_text(frame, f"PATIENT: {rx.patient_name}", (20, CFG.DISPLAY_SIZE[1]-30), 0.7, C_WHITE, 2)

    # 2. Prescription Checklist (Left Side Logic)
    x_base = CFG.DISPLAY_SIZE[0] - 280
    y_pos = 50
    draw_text(frame, "RX CHECKLIST:", (x_base, 30), 0.6, C_YELLOW, 2)
    
    for norm, data in rx.target_drugs.items():
        is_done = data['found'] > 0 or data['manual']
        color = C_GREEN if is_done else C_GRAY
        icon = "✔" if is_done else "□"
        qty = f"x{data['qty']}" if data['qty'] else ""
        text = f"{icon} {data['original'][:15].upper()} {qty}"
        draw_text(frame, text, (x_base, y_pos), 0.55, color, 1)
        y_pos += 30

    # 3. WARNING: Extra Items (Show below checklist)
    if rx.detected_extras:
        y_pos += 10
        draw_text(frame, "⚠️ UNEXPECTED ITEMS:", (x_base, y_pos), 0.6, C_RED, 2)
        y_pos += 25
        for extra in list(rx.detected_extras)[-3:]: # Show last 3
            draw_text(frame, f"• {extra.upper()}", (x_base, y_pos), 0.5, C_RED, 1)
            y_pos += 25

    # 4. Bounding Boxes
    with ai_proc.lock:
        for res in ai_proc.results:
            x1, y1, x2, y2 = res['box']
            conf = res['conf']
            
            if conf > CFG.VERIFY_THRESHOLD:
                if res['is_valid']:
                    # Correct Drug -> Green Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), C_GREEN, 2)
                    draw_text(frame, f"{res['label']}", (x1, y1-5), 0.5, C_GREEN, 2)
                else:
                    # Wrong Drug -> Red Box (Warning)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), C_RED, 2)
                    draw_text(frame, f"EXTRA: {res['label']}", (x1, y1-5), 0.5, C_RED, 2)

# ================= 🚀 MAIN =================
def main():
    try: camera = CameraHandler(width=CFG.DISPLAY_SIZE[0], height=CFG.DISPLAY_SIZE[1])
    except: return

    ai = AIProcessor().start()
    hn_queue = deque(["HN123", "HN456"]); current_hn = None

    print(f"🚀 Started in {CFG.MODE}. Controls: [N] Next | [Q] Quit")
    
    # Callback สำหรับ Mouse Click (เผื่ออยากกดติ๊กเอง)
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"🖱️ Clicked at {x},{y}")
            # Logic: ถ้าคลิกโซน Rx Checklist ให้ Toggle 'manual' (Mockup)
            if x > CFG.DISPLAY_SIZE[0] - 280:
                print("📝 Manual check triggered (Mockup)")

    window_name = "PillTrack Audit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        frame = camera.get_frame()
        if frame is None: time.sleep(0.1); continue
        
        ai.latest_frame = frame
        display_frame = frame.copy()
        
        if ai.rx.is_ready:
            draw_ui(display_frame, ai)
        else:
            draw_text(display_frame, "PRESS 'N' FOR PATIENT", (400, 360), 1.0, (255, 0, 0, 255), 2)

        # Show RGB via BGR conversion
        cv2.imshow(window_name, cv2.cvtColor(display_frame, cv2.COLOR_RGBA2BGR))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('n') and CFG.MODE == "integrated":
            hn_queue.rotate(-1); current_hn = hn_queue[0]
            print(f"⏩ Next: {current_hn}")
            ai.rx.reset()
            data = ai.his.fetch_prescription(current_hn)
            if data: ai.rx.update_from_his(data)

    camera.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()