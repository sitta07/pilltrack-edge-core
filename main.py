#!/usr/bin/env python3
"""
PILLTRACK – SENIOR EDITION (OPTIMIZED)
✔ Conditional Pipeline: YOLO -> DINO/SIFT only if detected
✔ Performance Optimizations:
  - Frame skipping for AI processing
  - Batch processing optimization
  - Memory-efficient operations
  - Reduced redundant computations
  - Optimized SIFT matching
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
    MODEL_PACK: str = yaml_cfg['artifacts']['model']
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
    AI_FRAME_SKIP: int = 2  # Process every Nth frame
    MAX_BATCH_SIZE: int = 8  # Max crops to process at once
    SIFT_TOP_K: int = 3  # Reduced from 5
    DINO_TOP_K: int = 5
    MIN_DINO_SCORE: float = 0.4
    VERIFY_THRESHOLD: float = 0.6
    
    # Display settings
    UI_UPDATE_FPS: int = 20  # UI refresh rate
    
    # Normalization constants
    MEAN: np.ndarray = field(default_factory=lambda: np.array([0.485, 0.456, 0.406], dtype=np.float32))
    STD: np.ndarray = field(default_factory=lambda: np.array([0.229, 0.224, 0.225], dtype=np.float32))

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ================= 🛠️ UTILS =================
def draw_text(img, text, pos, scale=0.5, color=(255,255,255), thickness=1):
    """Draw text with outline for better visibility"""
    cv2.putText(img, text, pos, FONT, scale, (0,0,0), thickness+2)
    cv2.putText(img, text, pos, FONT, scale, color, thickness)

def normalize_name(name: str) -> str:
    """Normalize drug name for comparison"""
    name = name.lower()
    name = re.sub(r'_pack.*', '', name)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

# ================= 🧠 PRESCRIPTION MANAGER =================
class PrescriptionManager:
    """Manages prescription drug list and verification"""
    
    def __init__(self):
        self.all_drugs = []
        self.norm_map = {}
        self.verified = set()
        self.load()

    def load(self):
        """Load drug list from JSON"""
        if not os.path.exists(CFG.DRUG_LIST_JSON):
            return
            
        with open(CFG.DRUG_LIST_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for d in data.get('drugs', []):
            norm = normalize_name(d)
            self.all_drugs.append(d.lower())
            self.norm_map[norm] = d.lower()

    def verify(self, detected_name: str) -> bool:
        """Verify if detected name matches prescription"""
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
    """Handles DINOv2 and SIFT feature extraction"""
    
    def __init__(self):
        print("⏳ Loading DINOv2...")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.model.eval().to(device)
        
        # Use half precision if GPU available for speed
        if device.type == 'cuda':
            self.model = self.model.half()
        
        self.sift = cv2.SIFT_create(nfeatures=500)  # Limit features for speed
        
        # Pre-allocate tensors for common batch sizes
        self.tensor_cache = {}

    def preprocess_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        """Batch preprocessing with optimizations"""
        batch = np.zeros((len(crop_list), 3, 224, 224), dtype=np.float32)
        
        for i, img in enumerate(crop_list):
            # Resize once
            img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            # Normalize in one go
            img_norm = (img_resized.astype(np.float32) / 255.0 - CFG.MEAN) / CFG.STD
            
            # Transpose
            batch[i] = img_norm.transpose(2, 0, 1)
        
        return batch

    @torch.no_grad()
    def extract_dino_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        """Extract DINOv2 embeddings for batch of crops"""
        if not crop_list:
            return np.array([])
        
        # Preprocess
        img_batch_np = self.preprocess_batch(crop_list)
        img_batch_t = torch.from_numpy(img_batch_np).to(device)
        
        # Use half precision if available
        if device.type == 'cuda':
            img_batch_t = img_batch_t.half()
        
        # Extract features
        embeddings = self.model(img_batch_t)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().float().numpy()

    def extract_sift(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Extract SIFT descriptors"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors

# ================= 🤖 AI PROCESSOR =================
class AIProcessor:
    """Main AI processing pipeline with optimizations"""
    
    def __init__(self):
        self.rx = PrescriptionManager()
        self.engine = FeatureEngine()
        
        # Database
        self.db_vectors = []
        self.db_names = []
        self.db_sift_map = {}
        self.bf = cv2.BFMatcher()
        self.load_db()
        
        # YOLO model
        print("⏳ Loading YOLO...")
        self.yolo = YOLO(CFG.MODEL_PACK)
        
        # Threading
        self.latest_frame = None
        self.results = []
        self.lock = threading.Lock()
        
        # Performance tracking
        self.ms = 0
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        
        # Frame skipping
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
        
        vectors = np.array(vectors, dtype=np.float32)
        # Normalize vectors สำหรับ Cosine Similarity
        faiss.normalize_L2(vectors)
        
        # --- [NEW: FAISS INDEX SETUP] ---
        dim = vectors.shape[1]
        # ใช้ IndexFlatIP (Inner Product) เพราะเรา Normalize แล้ว จะได้ค่าเดียวกับ Cosine Similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        # --------------------------------
        
        print(f"✅ FAISS Index Built: {len(vectors)} vectors")

    def get_sift_score(self, query_des: Optional[np.ndarray], 
                       target_des_list: List[np.ndarray]) -> float:
        """Calculate SIFT matching score (optimized)"""
        if query_des is None or not target_des_list:
            return 0.0
        
        max_score = 0.0
        
        for target_des in target_des_list:
            if target_des is None or len(target_des) < 2:
                continue
                
            try:
                # Use knnMatch with k=2 for ratio test
                matches = self.bf.knnMatch(query_des, target_des, k=2)
                
                # Lowe's ratio test
                good = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good.append(m)
                
                # Normalize score
                score = min(len(good) / CFG.SIFT_SATURATION, 1.0)
                max_score = max(max_score, score)
                
            except Exception as e:
                continue
        
        return max_score

    def process(self, frame: np.ndarray):
        """Process single frame through pipeline"""
        t0 = time.time()
        
        # STAGE 1: YOLO Detection
        img_resized = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE), 
                                 interpolation=cv2.INTER_LINEAR)
        
        res = self.yolo(img_resized, conf=CFG.CONF_THRESHOLD, verbose=False)[0]

        # Early exit if no detections
        if res.boxes is None or len(res.boxes) == 0:
            with self.lock:
                self.results = []
                self.ms = (time.time() - t0) * 1000
            return

        # STAGE 2: Feature Extraction & Matching
        temp_results = []
        sx = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE
        sy = CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
        
        crops = []
        box_coords = []
        
        # Collect crops
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            
            # Scale to display size
            dx1 = int(x1 * sx)
            dy1 = int(y1 * sy)
            dx2 = int(x2 * sx)
            dy2 = int(y2 * sy)
            
            # Extract crop with bounds checking
            crop = frame[max(0, dy1):min(frame.shape[0], dy2), 
                        max(0, dx1):min(frame.shape[1], dx2)]
            
            if crop.size > 0:
                crops.append(crop)
                box_coords.append([dx1, dy1, dx2, dy2])
        
       
        # ค้นหาในเมธอด process ช่วง Stage 2: Feature Extraction & Matching

        if crops:
            batch_dino = self.engine.extract_dino_batch(crops)
            scores, indices = self.index.search(batch_dino, k=CFG.DINO_TOP_K)
            
            for i, crop in enumerate(crops):
                sim_scores = scores[i]
                top_k_indices = indices[i]
                
                # --- [SENIOR OPTIMIZATION: EARLY SKIP] ---
                # ถ้าตัวที่เหมือนที่สุด (Top 1) ยังมีคะแนนต่ำกว่าเกณฑ์พื้นฐาน 
                # ไม่ต้องทำ SIFT ให้เสียเวลา ข้าม Crop นี้ไปเลย!
                if np.max(sim_scores) < CFG.MIN_DINO_SCORE:
                    continue

                best_label = "Unknown"
                max_fusion = 0.0
                seen_names = set()
                
                # ตัวแปรเก็บ SIFT Descriptor (ดึงแค่ครั้งเดียวต่อ 1 Crop ถ้าเจอ Candidate ที่ผ่านเกณฑ์)
                q_des = None 
                
                for idx_in_top_k, db_idx in enumerate(top_k_indices):
                    if db_idx == -1: continue
                    
                    name = self.db_names[db_idx]
                    if name in seen_names: continue
                    seen_names.add(name)
                    
                    dino_score = sim_scores[idx_in_top_k]
                    
                    # --- [SPECIFIC CANDIDATE SKIP] ---
                    # ถ้า DINO Score ของ candidate ตัวนี้ต่ำกว่า 0.5 ไม่ต้องเอาไปคิด SIFT Fusion
                    if dino_score < 0.5:
                        # แต่ถ้าอยากให้มันยังติดอันดับ "Unknown" หรือคะแนนต่ำๆ ไว้ ก็แค่ continue
                        continue 
                    
                    # สกัด SIFT เฉพาะตอนที่มี Candidate ผ่านเกณฑ์ DINO > 0.5 เท่านั้น
                    if q_des is None:
                        q_des = self.engine.extract_sift(crop)
                    
                    sift_score = self.get_sift_score(q_des, self.db_sift_map.get(name, []))
                    fusion_score = (dino_score * CFG.W_DINO) + (sift_score * CFG.W_SIFT)
                    
                    if fusion_score > max_fusion:
                        max_fusion = fusion_score
                        best_label = name
                
                # Store result
                temp_results.append({
                    'box': box_coords[i],
                    'label': best_label,
                    'conf': max_fusion
                })
                
                # Verify against prescription
                if max_fusion > CFG.VERIFY_THRESHOLD:
                    self.rx.verify(best_label)
                
                print(f"📡 [TRACKING] {best_label} ({max_fusion:.2f})")
        
        # Update results
        elapsed = (time.time() - t0) * 1000
        with self.lock:
            self.results = temp_results
            self.ms = elapsed
            self.fps_history.append(1000.0 / elapsed if elapsed > 0 else 0)

    def start(self):
        """Start processing thread"""
        threading.Thread(target=self.loop, daemon=True).start()
        return self

    def loop(self):
        """Main processing loop with frame skipping"""
        while True:
            # Frame skipping for performance
            self.process_counter += 1
            
            if self.process_counter >= CFG.AI_FRAME_SKIP:
                self.process_counter = 0
                
                if self.latest_frame is not None:
                    with self.lock:
                        work_frame = self.latest_frame.copy()
                    
                    self.process(work_frame)
            
            time.sleep(0.001)  # Small sleep to prevent CPU spinning

# ================= 🖥️ UI & DISPLAY =================
def draw_ui(frame: np.ndarray, ai_proc: AIProcessor):
    """Draw UI overlay on frame"""
    
    # Draw checklist
    rx = ai_proc.rx
    y_pos = 40
    
    for drug in rx.all_drugs:
        is_verified = drug in rx.verified
        color = (0, 255, 0) if is_verified else (180, 180, 180)
        text = f"✔ {drug.upper()}" if is_verified else f"□ {drug.upper()}"
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, FONT, 0.55, 2)
        
        # Draw text
        x_pos = CFG.DISPLAY_SIZE[0] - text_width - 10
        draw_text(frame, text, (x_pos, y_pos), 0.55, color)
        
        # Draw strikethrough if verified
        if is_verified:
            cv2.line(frame, (x_pos, y_pos - 8), 
                    (CFG.DISPLAY_SIZE[0] - 10, y_pos - 8), 
                    (0, 255, 0), 2)
        
        y_pos += 28

    # Draw detection boxes
    with ai_proc.lock:
        current_results = ai_proc.results.copy()
        latency = ai_proc.ms
        avg_fps = np.mean(ai_proc.fps_history) if ai_proc.fps_history else 0

    # for result in current_results:
    #     x1, y1, x2, y2 = result['box']
    #     label = result['label'].upper()
    #     confidence = result['conf']
        
    #     # Color based on confidence
    #     color = (0, 255, 0) if confidence > CFG.VERIFY_THRESHOLD else (0, 255, 255)
        
    #     # Draw box
    #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
    #     # Draw label background
    #     tag = f"{label} {confidence:.2f}"
    #     (tag_width, tag_height), _ = cv2.getTextSize(tag, FONT, 0.4, 1)
    #     cv2.rectangle(frame, (x1, y1 - tag_height - 10), 
    #                  (x1 + tag_width, y1), color, -1)
        
    #     # Draw label text
    #     cv2.putText(frame, tag, (x1, y1 - 5), FONT, 0.4, (0, 0, 0), 1)

    # Draw performance stats
    draw_text(frame, f"AI: {latency:.1f}ms | FPS: {avg_fps:.1f}", 
             (10, 20), 0.5, (0, 255, 255))

# ================= 🚀 MAIN =================
def main():
    """Main entry point"""
    
    # Sync manager
    if SyncManager:
        try:
            SyncManager().sync()
        except Exception as e:
            print(f"⚠️ Sync failed: {e}")

    # Initialize camera
    try:
        from picamera2 import Picamera2
        print("📷 Using Picamera2")
        
        cam_obj = Picamera2()
        config = cam_obj.create_preview_configuration(
            main={"size": CFG.DISPLAY_SIZE, "format": "RGB888"}
        )
        cam_obj.configure(config)
        cam_obj.start()
        
        def get_frame():
            return cam_obj.capture_array()
            
    except ImportError:
        print("📷 Using OpenCV camera")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CFG.DISPLAY_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG.DISPLAY_SIZE[1])
        
        def get_frame():
            ret, frame = cap.read()
            return frame if ret else None

    # Initialize AI processor
    ai = AIProcessor().start()
    
    # Setup window
    window_name = "PillTrack"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CFG.DISPLAY_SIZE[0], CFG.DISPLAY_SIZE[1])

    # Main loop
    last_ui_time = 0
    ui_interval = 1.0 / CFG.UI_UPDATE_FPS
    
    print("🚀 Starting main loop...")
    
    while True:
        # Capture frame
        frame = get_frame()
        if frame is None:
            continue

        # Update latest frame for AI processing
        with ai.lock:
            ai.latest_frame = frame

        # Update UI at specified FPS
        current_time = time.time()
        if current_time - last_ui_time > ui_interval:
            display_frame = frame.copy()
            draw_ui(display_frame, ai)
            cv2.imshow(window_name, display_frame)
            last_ui_time = current_time

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cv2.destroyAllWindows()
    print("👋 Shutting down...")

if __name__ == "__main__":
    main()