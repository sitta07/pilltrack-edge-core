#!/usr/bin/env python3
"""
PILLTRACK – SENIOR EDITION (ULTRA OPTIMIZED)
✔ Inference Engine: OpenVINO (Converted from YOLOv8/v11)
✔ Profiling: Data-driven bottleneck detection
✔ Performance: Frame skipping & Batch processing
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
    # ชี้ไปยัง Folder OpenVINO ที่ export มา
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
    UI_UPDATE_FPS: int = 20  
    
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
        img_batch_t = torch.from_numpy(self.preprocess_batch(crop_list)).to(device)
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
        self.db_names = []
        self.db_sift_map = {}
        self.bf = cv2.BFMatcher()
        self.load_db()
        
        print(f"⏳ Loading YOLO OpenVINO from: {CFG.MODEL_PACK}")
        # โหลด Folder OpenVINO ตรงๆ
        self.yolo = YOLO(CFG.MODEL_PACK, task="detect")
        
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
        print(f"✅ Database Loaded: {len(vectors)} vectors")

    def get_sift_score(self, query_des, target_des_list) -> float:
        if query_des is None or not target_des_list: return 0.0
        max_score = 0.0
        for target_des in target_des_list:
            if target_des is None or len(target_des) < 2: continue
            try:
                matches = self.bf.knnMatch(query_des, target_des, k=2)
                good = [m for m_pair in matches if len(m_pair) == 2 and m_pair[0].distance < 0.75 * m_pair[1].distance]
                max_score = max(max_score, min(len(good) / CFG.SIFT_SATURATION, 1.0))
            except: continue
        return max_score

    def process(self, frame: np.ndarray):
        t_start = time.perf_counter()
        prof = {}

        # 1. YOLO
        t0 = time.perf_counter()
        img_resized = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        res = self.yolo(img_resized, conf=CFG.CONF_THRESHOLD, verbose=False)[0]
        prof['yolo'] = (time.perf_counter() - t0) * 1000

        if res.boxes is None or len(res.boxes) == 0:
            with self.lock:
                self.results, self.ms = [], (time.perf_counter() - t_start) * 1000
            return

        # 2. Crops
        t1 = time.perf_counter()
        temp_results = []
        sx, sy = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE, CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
        crops, box_coords = [], []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            dx1, dy1, dx2, dy2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
            crop = frame[max(0,dy1):min(frame.shape[0],dy2), max(0,dx1):min(frame.shape[1],dx2)]
            if crop.size > 0:
                crops.append(crop)
                box_coords.append([dx1, dy1, dx2, dy2])
        prof['crop'] = (time.perf_counter() - t1) * 1000

        # 3. DINO & SIFT Fusion
        t2 = time.perf_counter()
        if crops:
            batch_dino = self.engine.extract_dino_batch(crops)
            scores, indices = self.index.search(batch_dino, k=CFG.DINO_TOP_K)
            
            for i, crop in enumerate(crops):
                if np.max(scores[i]) < CFG.MIN_DINO_SCORE: continue
                best_label, max_f, seen, q_des = "Unknown", 0.0, set(), None
                for idx, db_idx in enumerate(indices[i]):
                    if db_idx == -1: continue
                    name = self.db_names[db_idx]
                    if name in seen: continue
                    seen.add(name)
                    d_score = scores[i][idx]
                    if d_score > 0.5:
                        if q_des is None: q_des = self.engine.extract_sift(crop)
                        s_score = self.get_sift_score(q_des, self.db_sift_map.get(name, []))
                        f_score = (d_score * CFG.W_DINO) + (s_score * CFG.W_SIFT)
                        if f_score > max_f: max_f, best_label = f_score, name
                
                temp_results.append({'box': box_coords[i], 'label': best_label, 'conf': max_f})
                if max_f > CFG.VERIFY_THRESHOLD: self.rx.verify(best_label)
        prof['match'] = (time.perf_counter() - t2) * 1000

        total_ms = (time.perf_counter() - t_start) * 1000
        print(f"📊 [YOLO: {prof['yolo']:.1f}ms] [Match: {prof.get('match',0):.1f}ms] Total: {total_ms:.1f}ms")
        
        with self.lock:
            self.results, self.ms = temp_results, total_ms
            self.fps_history.append(1000.0/total_ms if total_ms > 0 else 0)

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def loop(self):
        while True:
            self.process_counter += 1
            if self.process_counter >= CFG.AI_FRAME_SKIP and self.latest_frame is not None:
                self.process_counter = 0
                with self.lock: frame = self.latest_frame.copy()
                self.process(frame)
            time.sleep(0.001)

# ================= 🖥️ UI & DISPLAY =================
def draw_ui(frame, ai):
    rx = ai.rx
    y_pos = 40
    for drug in rx.all_drugs:
        v = drug in rx.verified
        c = (0, 255, 0) if v else (180, 180, 180)
        t = f"✔ {drug.upper()}" if v else f"□ {drug.upper()}"
        w, _ = cv2.getTextSize(t, FONT, 0.55, 2)[0]
        draw_text(frame, t, (CFG.DISPLAY_SIZE[0]-w-10, y_pos), 0.55, c)
        if v: cv2.line(frame, (CFG.DISPLAY_SIZE[0]-w-10, y_pos-8), (CFG.DISPLAY_SIZE[0]-10, y_pos-8), (0,255,0), 2)
        y_pos += 28

    with ai.lock:
        res, ms, fps = ai.results.copy(), ai.ms, np.mean(ai.fps_history) if ai.fps_history else 0
    
    # Optional: Uncomment to draw boxes
    # for r in res:
    #     cv2.rectangle(frame, (r['box'][0], r['box'][1]), (r['box'][2], r['box'][3]), (0,255,0), 2)

    draw_text(frame, f"AI: {ms:.1f}ms | FPS: {fps:.1f}", (10, 20), 0.5, (0, 255, 255))

def main():
    if SyncManager:
        try: SyncManager().sync()
        except: pass

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.DISPLAY_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.DISPLAY_SIZE[1])

    ai = AIProcessor().start()
    win = "PillTrack Senior"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    
    last_ui = 0
    while True:
        ret, frame = cap.read()
        if not ret: continue
        with ai.lock: ai.latest_frame = frame
        
        if time.time() - last_ui > (1.0/CFG.UI_UPDATE_FPS):
            disp = frame.copy()
            draw_ui(disp, ai)
            cv2.imshow(win, disp)
            last_ui = time.time()
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()

if __name__ == "__main__": main()