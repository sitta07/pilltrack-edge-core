#!/usr/bin/env python3
"""
PILLTRACK – SENIOR EDITION (TARGETED SEARCH & AUTO-COMPLETE)
✔ Optimization: Searches ONLY for prescribed drugs (Reduced Search Space)
✔ Automation: Auto-complete & Reset when prescription is filled
✔ Profiling: Granular timing logs (YOLO/DINO/SIFT)
✔ Pipeline: 100% RGB888
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
    
    MODE: str = yaml_cfg.get('operation', {}).get('mode', 'standalone')
    
    # Scoring
    W_DINO: float = 0.6
    W_SIFT: float = 0.4
    SIFT_SATURATION: int = 400
    
    # Performance
    AI_FRAME_SKIP: int = 2
    MIN_DINO_SCORE: float = 0.4
    VERIFY_THRESHOLD: float = 0.65 # Increased slightly for targeted search
    
    # Normalization (RGB)
    MEAN: np.ndarray = field(default_factory=lambda: np.array([0.485, 0.456, 0.406], dtype=np.float32))
    STD: np.ndarray = field(default_factory=lambda: np.array([0.229, 0.224, 0.225], dtype=np.float32))

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ================= 📷 CAMERA HANDLER (RGB Strict) =================
class CameraHandler:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.use_picamera = False
        self.cap = None
        self.picam = None
        
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            config = self.picam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            self.picam.configure(config)
            self.picam.start()
            self.use_picamera = True
            print("📷 Camera: Using Picamera2 (RGB888 Native)")
        except Exception as e:
            print(f"⚠️ Picamera2 failed. Falling back to OpenCV. {e}")
            self.use_picamera = False
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_frame(self):
        if self.use_picamera:
            return self.picam.capture_array()
        else:
            ret, frame = self.cap.read()
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

    def release(self):
        if self.use_picamera: self.picam.stop()
        elif self.cap: self.cap.release()

# ================= 🧠 PRESCRIPTION MANAGER =================
class PrescriptionManager:
    def __init__(self):
        self.target_drugs = {} # {norm_name: {original, qty, found}}
        self.patient_name = "Standalone"
        self.is_ready = False
        self.is_completed = False
        self.complete_timestamp = 0
        
        # Standalone mode: Load everything
        if CFG.MODE == "standalone":
            self.load_local_all()
            self.is_ready = True

    def load_local_all(self):
        if not os.path.exists(CFG.DRUG_LIST_JSON): return
        with open(CFG.DRUG_LIST_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for d in data.get('drugs', []):
            self.target_drugs[normalize_name(d)] = {"original": d, "qty": None, "found": 0}

    def update_from_his(self, his_data: Dict):
        self.reset() # Clear old data
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
        print(f"📦 Loaded Prescription for {self.patient_name}")

    def verify(self, detected_name: str):
        if self.is_completed: return False
        
        norm_det = normalize_name(detected_name)
        if norm_det in self.target_drugs:
            self.target_drugs[norm_det]['found'] = 1
            self.check_complete()
            return True
        return False

    def check_complete(self):
        # Check if all drugs are found
        all_found = all(d['found'] > 0 for d in self.target_drugs.values())
        if all_found and not self.is_completed:
            self.is_completed = True
            self.complete_timestamp = time.time()
            print(f"✅ PRESCRIPTION COMPLETED FOR {self.patient_name}!")

    def reset(self):
        self.target_drugs = {}
        self.patient_name = "Waiting..."
        self.is_ready = False
        self.is_completed = False
        self.complete_timestamp = 0

# ================= 🛠️ UTILS =================
def normalize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'_pack.*', '', name)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def draw_text(img, text, pos, scale=0.5, color=(255,255,255), thickness=1):
    cv2.putText(img, text, pos, FONT, scale, (0,0,0), thickness+2)
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
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors

# ================= 🤖 AI PROCESSOR =================
class AIProcessor:
    def __init__(self):
        self.rx = PrescriptionManager()
        self.engine = FeatureEngine()
        self.his = HISConnector()
        
        # Full Database
        self.full_db_vectors = {} # {norm_name: [vectors]}
        self.full_db_sift = {}    # {norm_name: sift_des}
        
        # Active Search Space (Subset for current patient)
        self.active_vectors = None # Matrix of vectors for current drugs
        self.active_names = []     # Corresponding names
        
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
        with open(CFG.DB_PACKS_VEC, 'rb') as f:
            raw = pickle.load(f)
        
        for name, data in raw.items():
            norm = normalize_name(name)
            dino_list = data.get('dino', []) if isinstance(data, dict) else data
            sift_list = data.get('sift', []) if isinstance(data, dict) else []
            
            self.full_db_vectors[norm] = dino_list
            self.full_db_sift[norm] = sift_list[:CFG.SIFT_TOP_K]
            
        print(f"✅ Database loaded: {len(self.full_db_vectors)} drugs available.")

    def prepare_search_space(self):
        """Builds a mini-database containing ONLY the patient's drugs"""
        if not self.rx.target_drugs: return
        
        vectors = []
        names = []
        
        for norm_name in self.rx.target_drugs.keys():
            if norm_name in self.full_db_vectors:
                # Add all vectors associated with this drug
                for vec in self.full_db_vectors[norm_name]:
                    vectors.append(vec)
                    names.append(norm_name)
        
        if vectors:
            self.active_vectors = np.array(vectors, dtype=np.float32) # (N, 384)
            # Transpose for fast dot product: (384, N)
            self.active_vectors = self.active_vectors.T 
            self.active_names = names
            print(f"🎯 Search Space Optimized: Monitoring {len(names)} variants of prescribed drugs only.")
        else:
            print("⚠️ Warning: Prescribed drugs not found in local database!")
            self.active_vectors = None

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
        # 0. Check Status
        if not self.rx.is_ready or self.rx.is_completed: return
        if self.active_vectors is None: return

        # Profiling Timers
        t_start = time.perf_counter()
        
        # --- 1. YOLO Detection ---
        t1 = time.perf_counter()
        img_resized = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        res = self.yolo(img_resized, conf=CFG.CONF_THRESHOLD, verbose=False)[0]
        t_yolo = (time.perf_counter() - t1) * 1000

        if res.boxes is None or len(res.boxes) == 0:
            with self.lock: self.results = []
            return

        # --- 2. Cropping ---
        t2 = time.perf_counter()
        sx, sy = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE, CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
        crops, box_coords = [], []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            dx1, dy1, dx2, dy2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
            crop = frame[max(0, dy1):min(frame.shape[0], dy2), max(0, dx1):min(frame.shape[1], dx2)]
            if crop.size > 0: crops.append(crop); box_coords.append([dx1, dy1, dx2, dy2])
        t_crop = (time.perf_counter() - t2) * 1000

        temp_results = []
        t_dino, t_match = 0, 0

        if crops:
            # --- 3. DINOv2 Extraction ---
            t3 = time.perf_counter()
            batch_dino = self.engine.extract_dino_batch(crops) # (B, 384)
            t_dino = (time.perf_counter() - t3) * 1000
            
            # --- 4. Targeted Matching (Dot Product) ---
            t4 = time.perf_counter()
            # Similarity = Batch (B, 384) @ ActiveDatabase (384, N) = (B, N)
            sim_matrix = np.dot(batch_dino, self.active_vectors)
            
            for i, crop in enumerate(crops):
                # Find best match in the reduced search space
                best_idx = np.argmax(sim_matrix[i])
                dino_score = sim_matrix[i][best_idx]
                
                # Filter weak matches
                if dino_score < CFG.MIN_DINO_SCORE: continue
                
                matched_name = self.active_names[best_idx]
                
                # Hybrid verification with SIFT (Only if DINO is promising)
                q_des = self.engine.extract_sift(crop)
                sift_score = self.get_sift_score(q_des, self.full_db_sift.get(matched_name, []))
                
                fusion = (dino_score * CFG.W_DINO) + (sift_score * CFG.W_SIFT)
                
                temp_results.append({
                    'box': box_coords[i], 
                    'label': self.rx.target_drugs[matched_name]['original'], 
                    'conf': fusion
                })
                
                if fusion > CFG.VERIFY_THRESHOLD:
                    self.rx.verify(matched_name)
            
            t_match = (time.perf_counter() - t4) * 1000

        with self.lock:
            self.results = temp_results

        # --- LOGGING ( Requirement 3 ) ---
        print(f"⏱️ [PROF] YOLO: {t_yolo:.1f}ms | Crop: {t_crop:.1f}ms | DINO: {t_dino:.1f}ms | Match: {t_match:.1f}ms")

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
    
    # 1. Status Bar
    status_color = (0, 255, 0) if rx.is_completed else (0, 255, 255)
    status_text = "COMPLETED - RESETTING..." if rx.is_completed else f"PATIENT: {rx.patient_name}"
    draw_text(frame, status_text, (20, CFG.DISPLAY_SIZE[1] - 30), 0.7, status_color, 2)

    # 2. Prescription List
    y_pos = 50
    draw_text(frame, "PRESCRIPTION:", (CFG.DISPLAY_SIZE[0] - 250, 30), 0.6, (255, 255, 0), 2)
    
    for norm, data in rx.target_drugs.items():
        is_found = data['found'] > 0
        color = (0, 255, 0) if is_found else (180, 180, 180)
        icon = "✔" if is_found else "□"
        text = f"{icon} {data['original'].upper()} x{data['qty']}"
        draw_text(frame, text, (CFG.DISPLAY_SIZE[0] - 240, y_pos), 0.5, color, 1)
        y_pos += 30

    # 3. Bounding Boxes
    if not rx.is_completed:
        with ai_proc.lock:
            for res in ai_proc.results:
                x1, y1, x2, y2 = res['box']
                color = (0, 255, 0) if res['conf'] > CFG.VERIFY_THRESHOLD else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_text(frame, f"{res['label']} {res['conf']:.2f}", (x1, y1-5), 0.4, color, 1)

# ================= 🚀 MAIN =================
def main():
    if SyncManager:
        try: SyncManager().sync()
        except: pass

    try: camera = CameraHandler(width=CFG.DISPLAY_SIZE[0], height=CFG.DISPLAY_SIZE[1])
    except: return

    ai = AIProcessor().start()
    
    print(f"🚀 Started in {CFG.MODE} mode (RGB888).")
    if CFG.MODE == "integrated":
        print("⌨️ Press 'H' to fetch prescription.")

    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        ai.latest_frame = frame
        display_frame = frame.copy()
        
        # UI Logic
        if ai.rx.is_ready:
            draw_ui(display_frame, ai)
            
            # Check for Auto-Reset
            if ai.rx.is_completed:
                if time.time() - ai.rx.complete_timestamp > 3.0: # Wait 3 seconds
                    print("🔄 Auto-resetting for next patient...")
                    ai.rx.reset()
                    ai.active_vectors = None # Clear search space
        else:
            draw_text(display_frame, "WAITING FOR PATIENT... (PRESS 'H')", (400, 360), 0.8, (255, 0, 0), 2)

        # Show (Swap to BGR for display only)
        cv2.imshow("PillTrack HIS", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('h') and CFG.MODE == "integrated":
            print("🔄 Fetching data...")
            data = ai.his.fetch_prescription("HN123")
            if data: 
                ai.rx.update_from_his(data)
                ai.prepare_search_space() # <--- สร้าง Mini Search Space ทันทีที่ได้ข้อมูล

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()