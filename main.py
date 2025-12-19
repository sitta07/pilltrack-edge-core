#!/usr/bin/env python3
"""
PILLTRACK – HYBRID ENGINE (OPTIMIZED BATCH + MATRIX)
✔ Detailed Console Logs (Human Readable)
✔ RGB8888 END-TO-END
✔ Checklist UI
✔ Batch Inference & Vectorized Search
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

from dataclasses import dataclass
from typing import Tuple, List, Dict
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
    
    # Preprocessing constants
    MEAN: np.ndarray = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD: np.ndarray = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

# ================= 🎨 FEATURE ENGINE (OPTIMIZED) =================
class FeatureEngine:
    def __init__(self):
        print("⏳ Loading DINOv2 (ViT-B/14)...")
        # Load model logic
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval().to(device)
        self.sift = cv2.SIFT_create()

    def preprocess_batch(self, crop_list):
        """Optimized numpy-based preprocessing instead of Torchvision transforms"""
        batch = []
        for img in crop_list:
            # Resize
            img = cv2.resize(img, (224, 224))
            # Normalize & Transpose (HWC -> CHW)
            img = (img.astype(np.float32) / 255.0 - CFG.MEAN) / CFG.STD
            img = img.transpose(2, 0, 1)
            batch.append(img)
        return np.array(batch)

    @torch.no_grad()
    def extract_dino_batch(self, crop_list):
        if not crop_list: return np.array([])
        
        # 1. Prepare Batch
        img_batch_np = self.preprocess_batch(crop_list)
        img_batch_t = torch.from_numpy(img_batch_np).to(device)
        
        # 2. Batch Inference (Fast!)
        embeddings = self.model(img_batch_t)
        
        # 3. Normalize (L2) directly on tensor
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()

    def extract_sift(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, sift_des = self.sift.detectAndCompute(gray, None)
        return sift_des

# ================= 🤖 AI PROCESSOR (VECTORIZED) =================
class AIProcessor:
    def __init__(self):
        self.rx = PrescriptionManager()
        self.engine = FeatureEngine()
        
        # Flatten Database for Matrix Multiplication
        self.db_vectors = []    # Matrix (N, 768)
        self.db_names = []      # List of names corresponding to rows
        self.db_sift_map = {}   # Dictionary for SIFT lookup
        
        self.bf = cv2.BFMatcher()
        self.load_db()
        
        print("⏳ Loading YOLO...")
        self.yolo = YOLO(CFG.MODEL_PACK)
        
        self.latest = None
        self.lock = threading.Lock()
        self.ms = 0
        self.last_log_time = 0 

    def load_db(self):
        if not os.path.exists(CFG.DB_PACKS_VEC):
            print(f"❌ Error: Database {CFG.DB_PACKS_VEC} not found")
            return

        with open(CFG.DB_PACKS_VEC,'rb') as f:
            raw = pickle.load(f)
            
        print(f"📦 Loading Database: {len(raw)} entries")
        
        # Flatten logic: Convert dict to big matrix
        for name, data in raw.items():
            dino_list = []
            if isinstance(data, dict):
                dino_list = data.get('dino', [])
                self.db_sift_map[name] = data.get('sift', [])
            elif isinstance(data, list):
                dino_list = data
                self.db_sift_map[name] = []
            
            for vec in dino_list:
                self.db_vectors.append(np.array(vec))
                self.db_names.append(name)
                
        self.db_vectors = np.array(self.db_vectors) # (Total_Samples, 768)
        # Normalize DB Matrix just in case
        norms = np.linalg.norm(self.db_vectors, axis=1, keepdims=True)
        self.db_vectors = self.db_vectors / (norms + 1e-8)
        
        print(f"⚡ Database Optimized: Matrix Shape {self.db_vectors.shape}")

    def get_sift_score(self, query_des, target_des_list):
        if query_des is None or not target_des_list: return 0.0
        max_score = 0.0
        # Check only first few reliable targets to save time if list is long
        for target_des in target_des_list[:5]: 
            if target_des is None: continue
            try:
                matches = self.bf.knnMatch(query_des, target_des, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                score = min(len(good) / CFG.SIFT_SATURATION, 1.0)
                if score > max_score: max_score = score
            except: continue
        return max_score

    def process(self, frame):
        t0 = time.time()
        
        # 1. YOLO Inference (Resize handled internally mostly, but explicit is safe)
        img_resized = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        res = self.yolo(img_resized, conf=CFG.CONF_THRESHOLD, verbose=False)[0]

        if res.boxes is None or len(res.boxes) == 0: 
            self.ms = (time.time()-t0)*1000
            return

        # 2. Collect Crops (Batch Preparation)
        crops = []
        valid_boxes = []
        
        sx = CFG.DISPLAY_SIZE[0]/CFG.AI_SIZE
        sy = CFG.DISPLAY_SIZE[1]/CFG.AI_SIZE

        for box in res.boxes:
            x1,y1,x2,y2 = box.xyxy[0].int().tolist()
            rx1,ry1,rx2,ry2 = int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
            
            # Boundary check
            h, w = frame.shape[:2]
            ry1, ry2 = max(0, ry1), min(h, ry2)
            rx1, rx2 = max(0, rx1), min(w, rx2)
            
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10: continue
            
            crops.append(crop)
            valid_boxes.append((rx1, ry1, rx2, ry2))

        if not crops: return

        # 3. DINO Batch Inference (Run ONCE for all pills)
        # Returns Matrix (Num_Pills, 768)
        batch_dino_vecs = self.engine.extract_dino_batch(crops)

        # 4. Matrix Similarity Search (Vectorized)
        # (Num_Pills, 768) @ (DB_Size, 768).T = (Num_Pills, DB_Size)
        similarity_matrix = np.dot(batch_dino_vecs, self.db_vectors.T)

        # Loop through each detected pill
        for i, crop in enumerate(crops):
            q_vec = batch_dino_vecs[i]
            sim_scores = similarity_matrix[i]
            
            # Get Top 5 candidates indices
            top_k_indices = np.argsort(sim_scores)[::-1][:5]
            
            # --- Candidate Refinement with SIFT ---
            best_final_score = -1.0
            best_label = "Unknown"
            log_dino = 0.0
            log_sift = 0.0
            
            # Compute SIFT only once per crop
            q_des = self.engine.extract_sift(crop)

            seen_names = set()
            
            for idx in top_k_indices:
                candidate_name = self.db_names[idx]
                dino_score = sim_scores[idx]
                
                # Skip duplicate name checks for same pill to save SIFT time
                if candidate_name in seen_names: continue
                seen_names.add(candidate_name)
                
                if dino_score < 0.4: continue # Early exit if DINO is low

                sift_score = self.get_sift_score(q_des, self.db_sift_map.get(candidate_name, []))
                final_score = (dino_score * CFG.W_DINO) + (sift_score * CFG.W_SIFT)

                if final_score > best_final_score:
                    best_final_score = final_score
                    best_label = candidate_name
                    log_dino = dino_score
                    log_sift = sift_score

            # --- LOGGING & DECISION ---
            if best_final_score > 0.60:
                self.rx.verify(best_label)
                
                # Visualize (Optional - minimal drawing here to save time)
                # Drawing handled by UI mostly, but we can log
                
                current_time = time.time()
                if (current_time - self.last_log_time > 1.0) and (best_final_score > 0.65):
                    self.last_log_time = current_time
                    status = "✅"
                    print(f"{status} [{best_label.upper()}] Conf: {best_final_score*100:.1f}% (D:{log_dino*100:.0f} S:{log_sift*100:.0f})")

        self.ms = (time.time()-t0)*1000

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def loop(self):
        while True:
            if self.latest is not None:
                # Use lock only when copying to prevent tearing, keep it brief
                with self.lock:
                    frame_to_process = self.latest.copy()
                self.process(frame_to_process)
            time.sleep(0.005) # Reduced sleep for faster response

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

    print("🚀 System Started (Optimized Engine). Press 'q' to exit.")

    while True:
        frame = cam.get()
        if frame is None: continue

        # Thread-safe update
        with ai.lock:
            ai.latest = frame
        
        # Draw UI on the MAIN thread copy (to avoid modifying ai.latest being read)
        display_frame = frame.copy()
        draw_ui(display_frame, ai.rx)
        cv2.putText(display_frame, f"AI: {ai.ms:.1f}ms", (10, 20), FONT, 0.5, (0,255,255), 1)
        cv2.imshow("PillTrack", display_frame)

        if cv2.waitKey(1) == ord('q'):
            break
            
    cv2.destroyAllWindows()