#!/usr/bin/env python3
"""
PILLTRACK – SENIOR EDITION (HIS INTEGRATED + ROBUST CAMERA)
✔ Supports Standalone & Integrated (HIS) Modes
✔ Auto-detects Raspberry Pi Camera (Picamera2) vs USB Webcam (OpenCV)
✔ Real-time Cross-check with Hospital Prescription
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

# --- [NEW IMPORT] ---
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
    
    # [NEW] Operation Settings
    MODE: str = yaml_cfg.get('operation', {}).get('mode', 'standalone')
    
    # Scoring weights
    W_DINO: float = 0.6
    W_SIFT: float = 0.4
    SIFT_SATURATION: int = 400
    
    # Performance settings
    AI_FRAME_SKIP: int = 2
    MAX_BATCH_SIZE: int = 8
    SIFT_TOP_K: int = 3
    DINO_TOP_K: int = 5
    MIN_DINO_SCORE: float = 0.4
    VERIFY_THRESHOLD: float = 0.6
    UI_UPDATE_FPS: int = 20
    
    MEAN: np.ndarray = field(default_factory=lambda: np.array([0.485, 0.456, 0.406], dtype=np.float32))
    STD: np.ndarray = field(default_factory=lambda: np.array([0.229, 0.224, 0.225], dtype=np.float32))

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ================= 📷 CAMERA HANDLER (NEW) =================
class CameraHandler:
    """
    Handles camera initialization safely.
    Prioritizes Picamera2 (for RPi 5) -> Falls back to OpenCV (for Mac/PC/USB)
    """
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.use_picamera = False
        self.cap = None
        self.picam = None
        
        # Try importing Picamera2
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            config = self.picam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam.configure(config)
            self.picam.start()
            self.use_picamera = True
            print("📷 Camera: Using Picamera2 (Libcamera)")
        except Exception as e:
            print(f"⚠️ Picamera2 not found or failed ({e}). Falling back to OpenCV.")
            self.use_picamera = False
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("❌ Could not open any camera (Picamera2 failed, OpenCV failed).")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            print("📷 Camera: Using OpenCV VideoCapture(0)")

    def get_frame(self):
        if self.use_picamera:
            # Picamera2 returns array directly (RGB usually, OpenCV needs BGR)
            frame = self.picam.capture_array()
            # Picamera2 often gives RGB, OpenCV needs BGR for imshow
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame

    def release(self):
        if self.use_picamera:
            self.picam.stop()
        elif self.cap:
            self.cap.release()

# ================= 🧠 PRESCRIPTION MANAGER =================
class PrescriptionManager:
    """Manages drug validation based on HIS or Local DB"""
    
    def __init__(self):
        self.target_drugs = {} # Format: {norm_name: {"original": name, "qty": x, "found": 0}}
        self.patient_name = "Standalone Mode"
        self.is_ready = False
        
        if CFG.MODE == "standalone":
            self.load_local_all()
            self.is_ready = True

    def load_local_all(self):
        """Load all drugs for search mode"""
        if not os.path.exists(CFG.DRUG_LIST_JSON): return
        with open(CFG.DRUG_LIST_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for d in data.get('drugs', []):
            self.target_drugs[normalize_name(d)] = {"original": d, "qty": None, "found": 0}

    def update_from_his(self, his_data: Dict):
        """Update target drugs from HIS API data"""
        self.target_drugs = {}
        self.patient_name = his_data.get('patient_name', 'Unknown')
        for item in his_data.get('prescription', []):
            norm = normalize_name(item['name'])
            self.target_drugs[norm] = {
                "original": item['name'],
                "qty": item['amount'],
                "found": 0
            }
        self.is_ready = True
        print(f"📦 Prescription Loaded for {self.patient_name}: {len(self.target_drugs)} items")

    def verify(self, detected_name: str):
        """Check if detected drug is in the target list"""
        norm_det = normalize_name(detected_name)
        if norm_det in self.target_drugs:
            # ในที่นี้เรานับว่า 'เจอแล้ว' (สามารถขยาย logic ไปนับจำนวนเม็ดได้)
            self.target_drugs[norm_det]['found'] = 1 
            return True
        return False

# ================= 🛠️ UTILS =================
def draw_text(img, text, pos, scale=0.5, color=(255,255,255), thickness=1):
    cv2.putText(img, text, pos, FONT, scale, (0,0,0), thickness+2)
    cv2.putText(img, text, pos, FONT, scale, color, thickness)

def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'_pack.*', '', name)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

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
            img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors

# ================= 🤖 AI PROCESSOR =================
class AIProcessor:
    def __init__(self):
        self.rx = PrescriptionManager()
        self.engine = FeatureEngine()
        self.his = HISConnector() # [NEW]
        
        self.db_vectors, self.db_names, self.db_sift_map = [], [], {}
        self.bf = cv2.BFMatcher()
        self.load_db()
        
        print("⏳ Loading YOLO...")
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
            dino_list = data.get('dino', []) if isinstance(data, dict) else data
            self.db_sift_map[name] = (data.get('sift', []) if isinstance(data, dict) else [])[:CFG.SIFT_TOP_K]
            for vec in dino_list:
                vectors.append(np.array(vec))
                self.db_names.append(name)
        vectors = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

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
        if not self.rx.is_ready: return # รอดึงข้อมูลคนไข้ก่อน
        
        t_start = time.perf_counter()
        img_resized = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        res = self.yolo(img_resized, conf=CFG.CONF_THRESHOLD, verbose=False)[0]

        if res.boxes is None or len(res.boxes) == 0:
            with self.lock: self.results = []; self.ms = (time.perf_counter() - t_start)*1000
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
            scores, indices = self.index.search(batch_dino, k=CFG.DINO_TOP_K)
            
            for i, crop in enumerate(crops):
                sim_scores, top_k_indices = scores[i], indices[i]
                if np.max(sim_scores) < CFG.MIN_DINO_SCORE: continue

                best_label, max_fusion, seen_names, q_des = "Unknown", 0.0, set(), None
                for idx, db_idx in enumerate(top_k_indices):
                    if db_idx == -1: continue
                    name = self.db_names[db_idx]
                    if name in seen_names: continue
                    seen_names.add(name)
                    
                    dino_score = sim_scores[idx]
                    if dino_score > 0.5:
                        if q_des is None: q_des = self.engine.extract_sift(crop)
                        sift_score = self.get_sift_score(q_des, self.db_sift_map.get(name, []))
                        fusion = (dino_score * CFG.W_DINO) + (sift_score * CFG.W_SIFT)
                        if fusion > max_fusion: max_fusion = fusion; best_label = name
                
                temp_results.append({'box': box_coords[i], 'label': best_label, 'conf': max_fusion})
                if max_fusion > CFG.VERIFY_THRESHOLD: self.rx.verify(best_label)

        with self.lock:
            self.results = temp_results
            self.ms = (time.perf_counter() - t_start) * 1000
            self.fps_history.append(1000.0 / self.ms if self.ms > 0 else 0)

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
    
    # Header
    status_text = f"MODE: {CFG.MODE.upper()} | PATIENT: {rx.patient_name}"
    draw_text(frame, status_text, (20, CFG.DISPLAY_SIZE[1] - 30), 0.6, (0, 255, 255), 2)

    # Checklist Area
    y_pos = 50
    draw_text(frame, "PRESCRIPTION CHECKLIST:", (CFG.DISPLAY_SIZE[0] - 300, 30), 0.6, (255, 255, 0), 2)
    
    for norm, data in rx.target_drugs.items():
        is_found = data['found'] > 0
        color = (0, 255, 0) if is_found else (150, 150, 150)
        qty_str = f" x{data['qty']}" if data['qty'] else ""
        text = f"{'✔' if is_found else '□'} {data['original'].upper()}{qty_str}"
        
        draw_text(frame, text, (CFG.DISPLAY_SIZE[0] - 280, y_pos), 0.5, color, 1)
        y_pos += 30

    # Boxes
    with ai_proc.lock:
        for res in ai_proc.results:
            x1, y1, x2, y2 = res['box']
            label, conf = res['label'], res['conf']
            color = (0, 255, 0) if conf > CFG.VERIFY_THRESHOLD else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_text(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 0.4, color, 1)

# ================= 🚀 MAIN =================
def main():
    if SyncManager:
        try: SyncManager().sync()
        except: pass

    # Initialize Camera Handler (Safe Mode)
    try:
        camera = CameraHandler(width=CFG.DISPLAY_SIZE[0], height=CFG.DISPLAY_SIZE[1])
    except RuntimeError as e:
        print(f"❌ FATAL ERROR: {e}")
        return

    ai = AIProcessor().start()
    
    print(f"🚀 Started in {CFG.MODE} mode.")
    if CFG.MODE == "integrated":
        print("⌨️ Press 'H' to fetch prescription for HN123")

    while True:
        frame = camera.get_frame()
        if frame is None:
            # เพิ่ม Log เพื่อแจ้งเตือนว่าอ่านภาพไม่ได้ (อาจจะแค่เฟรมกระตุก หรือหลุด)
            print("⚠️ Warning: Empty frame received. Retrying...")
            time.sleep(0.1)
            continue
        
        ai.latest_frame = frame
        display_frame = frame.copy()
        
        if ai.rx.is_ready:
            draw_ui(display_frame, ai)
        else:
            draw_text(display_frame, "WAITING FOR PATIENT DATA... (PRESS 'H')", (400, 360), 0.8, (0, 0, 255), 2)

        cv2.imshow("PillTrack HIS", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('h') and CFG.MODE == "integrated":
            # จำลองการส่ง HN123 ไปดึงข้อมูล
            print("🔄 Fetching data...")
            data = ai.his.fetch_prescription("HN123")
            if data: ai.rx.update_from_his(data)

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()