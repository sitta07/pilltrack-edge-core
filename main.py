#!/usr/bin/env python3
"""
PILLTRACK – GLOBAL SEARCH + PERFORMANCE LOGGER
✔ Pipeline: RGB8888 (RGBA 32-bit)
✔ Global Search: Scans against entire 3,000+ drug database
✔ Profiler: Logs ms for YOLO, DINO, SIFT, and Search individually
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
import onnxruntime as ort

# --- IMPORT MODULES ---
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
    W_DINO: float = 0.85
    W_SIFT: float = 0.15
    SIFT_SATURATION: int = 350
    
    SIFT_TOP_K: int = 3
    
    # Performance
    AI_FRAME_SKIP: int = 4
    MIN_DINO_SCORE: float = 0.45
    VERIFY_THRESHOLD: float = 0.65 
    
    # Normalization (RGB based)
    MEAN: np.ndarray = field(default_factory=lambda: np.array([0.485, 0.456, 0.406], dtype=np.float32))
    STD: np.ndarray = field(default_factory=lambda: np.array([0.229, 0.224, 0.225], dtype=np.float32))

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ================= 📷 CAMERA HANDLER =================
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
                main={"size": (self.width, self.height), "format": "XRGB8888"}
            )
            self.picam.configure(config)
            self.picam.start()
            self.use_picamera = True
            print("📷 Camera: Using Picamera2 (XRGB8888)")
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
            if not ret: return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    def release(self):
        if self.use_picamera: self.picam.stop()
        elif self.cap: self.cap.release()

# ================= 🧠 PRESCRIPTION MANAGER =================
class PrescriptionManager:
    def __init__(self):
        self.target_drugs = {} 
        self.patient_name = "Standalone"
        self.is_ready = False
        self.is_completed = False
        self.complete_timestamp = 0
        
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
        self.reset()
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

def draw_text(img, text, pos, scale=0.5, color=(255,255,255,255), thickness=1):
    black = (0, 0, 0, 255)
    cv2.putText(img, text, pos, FONT, scale, black, thickness+2)
    cv2.putText(img, text, pos, FONT, scale, color, thickness)

class FeatureEngine:
    def __init__(self):
        print("⏳ Loading DINOv2 (ONNX Runtime)...")
        # ใช้ ONNX แทน PyTorch!
        # providers=['CPUExecutionProvider'] คือรันบน CPU แต่เร็วกว่ามาก
        try:
            self.sess = ort.InferenceSession("dinov2_vits14.onnx", providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"❌ Error loading ONNX: {e}")
            print("💡 Did you run 'export_dino.py' yet?")
            raise e
            
        self.sift = cv2.SIFT_create(nfeatures=500)
        self.input_name = self.sess.get_inputs()[0].name
        
    def preprocess_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        batch = np.zeros((len(crop_list), 3, 224, 224), dtype=np.float32)
        for i, img in enumerate(crop_list):
            img_rgb = img[:, :, :3] 
            img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            img_norm = (img_resized.astype(np.float32) / 255.0 - CFG.MEAN) / CFG.STD
            batch[i] = img_norm.transpose(2, 0, 1)
        return batch

    def extract_dino_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        if not crop_list: return np.array([])
        
        # Preprocess
        img_batch_np = self.preprocess_batch(crop_list)
        
        # Run ONNX Inference
        # ไม่ต้องใช้ torch.no_grad() หรือ .to(device) เพราะ ONNX จัดการเอง
        outputs = self.sess.run(None, {self.input_name: img_batch_np})
        
        # Normalize (ทำด้วย Numpy เพราะ ONNX คืนค่าเป็น Numpy)
        embeddings = outputs[0]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-6) # ป้องกันหาร 0
        
        return embeddings

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
        
        self.full_db_vectors = {} 
        self.full_db_sift = {}
        
        # GLOBAL SEARCH Active Vectors
        self.active_vectors = None
        self.active_names = []
        
        self.bf = cv2.BFMatcher()
        self.load_db()
        self.prepare_global_search_space()
        
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

    def prepare_global_search_space(self):
        if not self.full_db_vectors: return
        vectors, names = [], []
        for norm_name, vec_list in self.full_db_vectors.items():
            for vec in vec_list:
                vectors.append(vec)
                names.append(norm_name)
        if vectors:
            self.active_vectors = np.array(vectors, dtype=np.float32).T 
            self.active_names = names
            print(f"🌍 Global Search Active: Scanning against {len(names)} vectors.")
        else:
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
        # ถ้าไม่มีใบยา หรือทำงานเสร็จแล้ว หรือยังไม่โหลด Database -> ไม่ต้องทำ
        if not self.rx.is_ready or self.rx.is_completed: return
        if self.active_vectors is None: return

        # --- TIMER START ---
        t_start_total = time.perf_counter()

        # 1. YOLO Detection
        t_yolo_start = time.perf_counter()
        img_rgb = frame[:, :, :3]
        img_resized = cv2.resize(img_rgb, (CFG.AI_SIZE, CFG.AI_SIZE))
        res = self.yolo(img_resized, conf=CFG.CONF_THRESHOLD, verbose=False)[0]
        t_yolo = (time.perf_counter() - t_yolo_start) * 1000

        # ถ้าไม่เจอยาเลย
        if res.boxes is None or len(res.boxes) == 0:
            with self.lock: self.results = []
            # print(f"⏱️ [IDLE] YOLO: {t_yolo:.1f}ms | No pills detected.") # ปิด Log รกๆ ได้ถ้าต้องการ
            return

        # 2. Cropping
        t_crop_start = time.perf_counter()
        sx, sy = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE, CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
        crops, box_coords = [], []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            dx1, dy1, dx2, dy2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
            # กันเหนียว: ตัดขอบไม่ให้เกินขนาดภาพ
            crop = frame[max(0, dy1):min(frame.shape[0], dy2), max(0, dx1):min(frame.shape[1], dx2)]
            if crop.size > 0: 
                crops.append(crop)
                box_coords.append([dx1, dy1, dx2, dy2])
        t_crop = (time.perf_counter() - t_crop_start) * 1000

        temp_results = []
        t_dino = 0
        t_search = 0
        t_sift_accum = 0 

        if crops:
            # 3. DINOv2 Extraction (ONNX)
            t_dino_start = time.perf_counter()
            batch_dino = self.engine.extract_dino_batch(crops) 
            t_dino = (time.perf_counter() - t_dino_start) * 1000
            
            # 4. Global Search & Verification
            t_search_start = time.perf_counter()
            
            # Vector Search (Dot Product)
            sim_matrix = np.dot(batch_dino, self.active_vectors)
            
            for i, crop in enumerate(crops):
                best_idx = np.argmax(sim_matrix[i])
                dino_score = sim_matrix[i][best_idx]
                
                # ❌ คอมเมนต์บรรทัดนี้ทิ้งไปก่อน (Disable Filter)
                # if dino_score < CFG.MIN_DINO_SCORE: 
                #     continue
                
                # ✅ เพิ่ม Print เพื่อดูว่าจริงๆ แล้วได้คะแนนเท่าไหร่?
                matched_name = self.active_names[best_idx]
                print(f"🧐 Debug: Box {i} matched '{matched_name}' with DINO score: {dino_score:.4f}")

                # ... (ส่วน SIFT เหมือนเดิม) ...
                
                # --- SIFT Verification ---
                t_sift_start = time.perf_counter()
                q_des = self.engine.extract_sift(crop)
                sift_score = self.get_sift_score(q_des, self.full_db_sift.get(matched_name, []))
                t_sift_accum += (time.perf_counter() - t_sift_start) * 1000
                # -------------------------
                
                # คำนวณคะแนนรวม
                fusion = (dino_score * CFG.W_DINO) + (sift_score * CFG.W_SIFT)
                
                # --- LOGIC การแสดงผล (แก้ไขใหม่) ---
                display_name = "Unknown"
                is_correct_drug = False
                
                if fusion > CFG.VERIFY_THRESHOLD:
                    # กรณี: มั่นใจว่าเป็นยานี้แน่ๆ
                    is_correct_drug = matched_name in self.rx.target_drugs
                    
                    if is_correct_drug:
                        # ถูกต้อง: เป็นยาในใบสั่ง
                        display_name = self.rx.target_drugs[matched_name]['original']
                        self.rx.verify(matched_name) # ติ๊กถูกในลิสต์
                    else:
                        # ผิด: เป็นยาที่มีใน Database แต่ไม่อยู่ในใบสั่ง
                        display_name = matched_name.upper()
                else:
                    # กรณี: ไม่มั่นใจ (คะแนนต่ำ) -> ให้โชว์เครื่องหมาย ? พร้อมคะแนน
                    display_name = f"? ({fusion:.2f})"
                    is_correct_drug = False # ถือว่าผิดไว้ก่อน
                
                # Append ลงผลลัพธ์เสมอ (เพื่อให้ UI วาดกรอบ)
                temp_results.append({
                    'box': box_coords[i], 
                    'label': display_name, 
                    'conf': fusion,
                    'is_correct': is_correct_drug
                })
            
            t_search = (time.perf_counter() - t_search_start) * 1000

        # อัปเดตผลลัพธ์ไปยัง Thread หลัก
        with self.lock:
            self.results = temp_results

        # --- FINAL LOGGING ---
        t_total = (time.perf_counter() - t_start_total) * 1000
        print(f"⏱️ TOTAL: {t_total:.1f}ms | YOLO: {t_yolo:.1f} | DINO: {t_dino:.1f} | SIFT: {t_sift_accum:.1f} | Search: {t_search:.1f}")
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
    
    COLOR_CYAN = (0, 255, 255, 255)
    COLOR_YELLOW = (255, 255, 0, 255)
    COLOR_GREEN = (0, 255, 0, 255)
    COLOR_GRAY = (150, 150, 150, 255)
    COLOR_RED = (0, 0, 255, 255)
    
    status_color = COLOR_GREEN if rx.is_completed else COLOR_CYAN
    status_text = "COMPLETED - RESETTING..." if rx.is_completed else f"PATIENT: {rx.patient_name}"
    draw_text(frame, status_text, (20, CFG.DISPLAY_SIZE[1] - 30), 0.7, status_color, 2)

    y_pos = 50
    draw_text(frame, "PRESCRIPTION:", (CFG.DISPLAY_SIZE[0] - 250, 30), 0.6, COLOR_YELLOW, 2)
    
    for norm, data in rx.target_drugs.items():
        is_found = data['found'] > 0
        color = COLOR_GREEN if is_found else COLOR_GRAY
        icon = "✔" if is_found else "□"
        text = f"{icon} {data['original'].upper()} x{data['qty']}"
        draw_text(frame, text, (CFG.DISPLAY_SIZE[0] - 240, y_pos), 0.5, color, 1)
        y_pos += 30

    if not rx.is_completed:
        with ai_proc.lock:
            for res in ai_proc.results:
                x1, y1, x2, y2 = res['box']
                if res['is_correct']:
                    color = COLOR_GREEN if res['conf'] > 0.8 else COLOR_CYAN
                else:
                    color = COLOR_RED 
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"{res['label']} {res['conf']:.2f}"
                if not res['is_correct']:
                    label_text = f"WRONG: {res['label']}"
                    
                draw_text(frame, label_text, (x1, y1-5), 0.4, color, 1)

# ================= 🚀 MAIN =================
# ================= 🚀 MAIN =================
def main():
    # 1. Sync Time (Optional)
    if SyncManager:
        try: SyncManager().sync()
        except: pass

    # 2. Init Camera
    try: 
        camera = CameraHandler(width=CFG.DISPLAY_SIZE[0], height=CFG.DISPLAY_SIZE[1])
    except Exception as e:
        print(f"❌ Camera Error: {e}")
        return

    # 3. Start AI
    ai = AIProcessor().start()
    
    # ------------------ 🔥 DYNAMIC QUEUE LOADER 🔥 ------------------
    # กำหนด Path ของไฟล์ Mock Data
    MOCK_DB_PATH = "mock_server/prescriptions.json" 
    hn_queue = deque()

    if os.path.exists(MOCK_DB_PATH):
        try:
            with open(MOCK_DB_PATH, 'r', encoding='utf-8') as f:
                mock_data = json.load(f)
            
            # ดึงเฉพาะ HN (Keys) มาสร้างเป็นคิว
            # เช่น ['HN123', 'HN456', 'HN789']
            hn_list = list(mock_data.keys())
            hn_queue = deque(hn_list)
            
            print(f"📂 Loaded {len(hn_list)} Patients from JSON: {hn_list}")
            
        except Exception as e:
            print(f"❌ JSON Error: {e} -> Fallback to dummy HN")
            hn_queue = deque(["HN123"])
    else:
        print(f"⚠️ File not found: {MOCK_DB_PATH} -> Using Hardcoded Fallback")
        hn_queue = deque(["HN123", "HN456"])
    # ---------------------------------------------------------------

    current_hn = None
    
    print(f"🚀 Started in {CFG.MODE} mode")
    print("⌨️  Controls: [N] Next Patient | [Q] Quit")

    # 4. Main Loop
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # ส่งเฟรมให้ AI (Thread แยกจะจัดการเอง)
        ai.latest_frame = frame
        display_frame = frame.copy()
        
        # --- UI LOGIC ---
        if ai.rx.is_ready:
            # ถ้ามีข้อมูลยา -> วาด UI ปกติ
            draw_ui(display_frame, ai)
            
            # ถ้าจ่ายยาครบแล้ว -> รอ 3 วิ แล้ว Reset อัตโนมัติ (หรือจะรอปุ่ม N ก็ได้แล้วแต่ดีไซน์)
            if ai.rx.is_completed:
                if time.time() - ai.rx.complete_timestamp > 3.0:
                    print("🔄 Completed! Auto-resetting state (Wait for Next Patient)...")
                    ai.rx.reset() # เคลียร์หน้าจอ รอคนกด N คนต่อไป
        else:
            # ถ้ายังไม่มีข้อมูลยา (ว่างเปล่า)
            status_text = f"NEXT: {hn_queue[0]}" if hn_queue else "NO DATA"
            draw_text(display_frame, f"PRESS 'N' FOR {status_text}", (380, 360), 0.8, (0, 255, 255, 255), 2)

        cv2.imshow("PillTrack HIS (Global Search)", display_frame)
        
        # --- CONTROL LOGIC ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('n'):
            if not hn_queue:
                print("⚠️ No more patients in queue!")
                continue

            # Rotate Queue (เอาคนแรกไปต่อท้าย หรือดึงออกมาเลยก็ได้)
            # ถ้าใช้ .rotate(-1) คือวนลูปไม่รู้จบ (HN123 -> HN456 -> HN123)
            hn_queue.rotate(-1)
            current_hn = hn_queue[0] 
            
            print(f"\n⏩ Switching to Patient: {current_hn}")
            
            # Reset สถานะเก่าก่อน
            ai.rx.reset()
            
            # เรียกข้อมูลจาก HISConnector (ซึ่งควรอ่านจากไฟล์เดียวกัน หรือ API)
            data = ai.his.fetch_prescription(current_hn)
            
            if data: 
                ai.rx.update_from_his(data)
            else:
                print(f"❌ Failed to fetch data for {current_hn} (Check HN in JSON)")

    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    print("👋 Exiting PillTrack...")

if __name__ == "__main__":
    main()