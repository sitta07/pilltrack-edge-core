#!/usr/bin/env python3
"""
PILLTRACK – SENIOR EDITION (DEBUGGED & OPTIMIZED)
✔ Fixed: Uncommented UI drawing loop (Visualization restored)
✔ Optimized: Fallback to 'Unknown' instead of skipping low-conf detections
✔ Performance: Frame skipping & Batch processing intact
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
    
    # Display settings
    UI_UPDATE_FPS: int = 30
    
    # Normalization constants
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

# ================= 🧠 PRESCRIPTION MANAGER =================
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

    def verify(self, detected_name: str) -> bool:
        norm_det = normalize_name(detected_name)
        
        for norm_drug, original in self.norm_map.items():
            if norm_det.startswith(norm_drug) or norm_drug.startswith(norm_det):
                if original not in self.verified:
                    print(f"🔥 [NEW VERIFIED]: {original.upper()}")
                    self.verified.add(original)
                return True
        return False

# ================= 🔍 FEATURE ENGINE =================
class FeatureEngine:
    def __init__(self):
        print("⏳ Loading DINOv2...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.eval().to(device)
        
        if device.type == 'cuda':
            self.model = self.model.half()
        
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
        if not crop_list:
            return np.array([])
        
        img_batch_np = self.preprocess_batch(crop_list)
        img_batch_t = torch.from_numpy(img_batch_np).to(device)
        
        if device.type == 'cuda':
            img_batch_t = img_batch_t.half()
        
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
        
        self.db_vectors = []
        self.db_names = []
        self.db_sift_map = {}
        self.bf = cv2.BFMatcher()
        self.load_db()
        
        print(f"⏳ Loading YOLO from: {CFG.MODEL_PACK}")
        try:
            # ถ้า CFG.MODEL_PACK เป็น folder มันจะพยายามหา .xml ในนั้น
            # ถ้าเป็นไฟล์ .xml หรือ .pt ก็จะโหลดตามปกติ
            self.yolo = YOLO(CFG.MODEL_PACK, task="detect")
            print("✅ YOLO Loaded Successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            # Fallback for debugging
            # self.yolo = YOLO("yolov8n.pt") 
        
        self.latest_frame = None
        self.results = []
        self.lock = threading.Lock()
        
        self.ms = 0
        self.fps_history = deque(maxlen=30)
        self.process_counter = 0

    def load_db(self):
        if not os.path.exists(CFG.DB_PACKS_VEC):
            print("⚠️ Database not found!")
            return
            
        with open(CFG.DB_PACKS_VEC, 'rb') as f:
            raw = pickle.load(f)
        
        vectors = []
        for name, data in raw.items():
            dino_list = data.get('dino', []) if isinstance(data, dict) else data
            sift_list = data.get('sift', []) if isinstance(data, dict) else []
            self.db_sift_map[name] = sift_list[:CFG.SIFT_TOP_K]
            
            for vec in dino_list:
                vectors.append(np.array(vec))
                self.db_names.append(name)
        
        if vectors:
            vectors = np.array(vectors, dtype=np.float32)
            faiss.normalize_L2(vectors)
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
            print(f"✅ FAISS Index Built: {len(vectors)} vectors")
        else:
            print("⚠️ Database is empty!")

    def get_sift_score(self, query_des: Optional[np.ndarray], target_des_list: List[np.ndarray]) -> float:
        if query_des is None or not target_des_list:
            return 0.0
        
        max_score = 0.0
        for target_des in target_des_list:
            if target_des is None or len(target_des) < 2: continue
            try:
                matches = self.bf.knnMatch(query_des, target_des, k=2)
                good = [m for m_pair in matches if len(m_pair) == 2 
                        and m_pair[0].distance < 0.75 * m_pair[1].distance]
                score = min(len(good) / CFG.SIFT_SATURATION, 1.0)
                max_score = max(max_score, score)
            except: continue
        return max_score

    def process(self, frame: np.ndarray):
        t_start = time.perf_counter()
        
        # --- STAGE 1: YOLO Detection ---
        img_resized = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE), interpolation=cv2.INTER_LINEAR)
        # verbose=False reduces terminal spam
        res = self.yolo(img_resized, conf=CFG.CONF_THRESHOLD, verbose=False)[0]

        if res.boxes is None or len(res.boxes) == 0:
            with self.lock:
                self.results = []
                self.ms = (time.perf_counter() - t_start) * 1000
            return

        # --- STAGE 2: Cropping ---
        temp_results = []
        sx, sy = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE, CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
        crops, box_coords = [], []
        
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            dx1, dy1, dx2, dy2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            crop = frame[max(0, dy1):min(frame.shape[0], dy2), 
                         max(0, dx1):min(frame.shape[1], dx2)]
            if crop.size > 0:
                crops.append(crop)
                box_coords.append([dx1, dy1, dx2, dy2])

        # --- STAGE 3: DINOv2 & FAISS ---
        if crops:
            batch_dino = self.engine.extract_dino_batch(crops)
            scores, indices = self.index.search(batch_dino, k=CFG.DINO_TOP_K)
            
            # --- STAGE 4: Fusion Loop ---
            for i, crop in enumerate(crops):
                sim_scores = scores[i]
                top_k_indices = indices[i]
                
                # Default "Unknown" label if scores are low
                best_label = "Unknown"
                max_fusion = float(np.max(sim_scores)) # Start with DINO score
                
                # Only perform advanced matching if initial DINO score is decent
                if max_fusion >= CFG.MIN_DINO_SCORE:
                    seen_names = set()
                    q_des = None
                    
                    for idx_in_top_k, db_idx in enumerate(top_k_indices):
                        if db_idx == -1: continue
                        name = self.db_names[db_idx]
                        if name in seen_names: continue
                        seen_names.add(name)
                        
                        dino_score = sim_scores[idx_in_top_k]
                        
                        if dino_score > 0.5:
                            if q_des is None: q_des = self.engine.extract_sift(crop)
                            sift_score = self.get_sift_score(q_des, self.db_sift_map.get(name, []))
                            fusion = (dino_score * CFG.W_DINO) + (sift_score * CFG.W_SIFT)
                            
                            if fusion > max_fusion:
                                max_fusion = fusion
                                best_label = name
                
                # Append result regardless (so we can see "Unknown" boxes)
                temp_results.append({'box': box_coords[i], 'label': best_label, 'conf': max_fusion})
                
                if max_fusion > CFG.VERIFY_THRESHOLD and best_label != "Unknown":
                    self.rx.verify(best_label)
        
        total_ms = (time.perf_counter() - t_start) * 1000
        # print(f"Processing: {len(temp_results)} objects | Time: {total_ms:.1f}ms")
        
        with self.lock:
            self.results = temp_results
            self.ms = total_ms
            self.fps_history.append(1000.0 / total_ms if total_ms > 0 else 0)

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def loop(self):
        while True:
            self.process_counter += 1
            if self.process_counter >= CFG.AI_FRAME_SKIP:
                self.process_counter = 0
                if self.latest_frame is not None:
                    with self.lock:
                        work_frame = self.latest_frame.copy()
                    self.process(work_frame)
            time.sleep(0.001)

# ================= 🖥️ UI & DISPLAY =================
def draw_ui(frame: np.ndarray, ai_proc: AIProcessor):
    """Draw UI overlay on frame"""
    
    # 1. Draw Checklist (Prescription)
    rx = ai_proc.rx
    y_pos = 40
    for drug in rx.all_drugs:
        is_verified = drug in rx.verified
        color = (0, 255, 0) if is_verified else (180, 180, 180)
        text = f"✔ {drug.upper()}" if is_verified else f"□ {drug.upper()}"
        (text_width, text_height), _ = cv2.getTextSize(text, FONT, 0.55, 2)
        x_pos = CFG.DISPLAY_SIZE[0] - text_width - 10
        draw_text(frame, text, (x_pos, y_pos), 0.55, color)
        if is_verified:
            cv2.line(frame, (x_pos, y_pos - 8), (CFG.DISPLAY_SIZE[0] - 10, y_pos - 8), (0, 255, 0), 2)
        y_pos += 28

    # 2. Draw Detection Boxes (UNCOMMENTED & FIXED)
    with ai_proc.lock:
        current_results = ai_proc.results.copy()
        latency = ai_proc.ms
        avg_fps = np.mean(ai_proc.fps_history) if ai_proc.fps_history else 0

    for result in current_results:
        x1, y1, x2, y2 = result['box']
        label = result['label'].upper() if result['label'] else "UNKNOWN"
        confidence = result['conf']
        
        # Color: Green for Verified High Conf, Yellow for Low Conf/Unknown
        if confidence > CFG.VERIFY_THRESHOLD and label != "UNKNOWN":
            color = (0, 255, 0)
        else:
            color = (0, 255, 255) # Yellow for warnings/unknowns
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        tag = f"{label} {confidence:.2f}"
        (tag_width, tag_height), _ = cv2.getTextSize(tag, FONT, 0.4, 1)
        
        # Boundary check for text
        y_text = y1 - 10 if y1 - 10 > 10 else y1 + 10
        
        cv2.rectangle(frame, (x1, y_text - tag_height), (x1 + tag_width, y_text + 5), color, -1)
        cv2.putText(frame, tag, (x1, y_text), FONT, 0.4, (0, 0, 0), 1)

    # 3. Draw Performance Stats
    status_color = (0, 255, 0) if avg_fps > 10 else (0, 0, 255)
    draw_text(frame, f"AI Latency: {latency:.1f}ms | FPS: {avg_fps:.1f}", 
             (10, 20), 0.5, status_color)

# ================= 🚀 MAIN =================
def main():
    if SyncManager:
        try: SyncManager().sync()
        except Exception as e: print(f"⚠️ Sync warning: {e}")

    # Camera Setup
    try:
        from picamera2 import Picamera2
        print("📷 Using Picamera2 (RPi5 Mode)")
        cam_obj = Picamera2()
        config = cam_obj.create_preview_configuration(
            main={"size": CFG.DISPLAY_SIZE, "format": "RGB888"}
        )
        cam_obj.configure(config)
        cam_obj.start()
        def get_frame(): return cam_obj.capture_array()
            
    except ImportError:
        print("📷 Using Standard OpenCV Camera")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.DISPLAY_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.DISPLAY_SIZE[1])
        def get_frame():
            ret, frame = cap.read()
            return frame if ret else None

    ai = AIProcessor().start()
    
    window_name = "PillTrack Senior"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CFG.DISPLAY_SIZE[0], CFG.DISPLAY_SIZE[1])

    print(f"🚀 System Ready. Press 'q' to exit.")
    
    while True:
        frame = get_frame()
        if frame is None: continue

        # Send to AI
        with ai.lock:
            ai.latest_frame = frame

        # Draw UI
        display_frame = frame.copy()
        draw_ui(display_frame, ai)
        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("👋 Shutting down...")

if __name__ == "__main__":
    main()