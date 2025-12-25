import threading
import time
import pickle
import os
import cv2
import numpy as np
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR # <--- [NEW] 1. Import OCR
from src.utils.config import CFG
from src.utils.helpers import normalize_name
from src.ai.engine import FeatureEngine
from src.core.prescription import PrescriptionManager

class AIProcessor:
    def __init__(self):
        try:
            self.rx = PrescriptionManager()
            self.engine = FeatureEngine()
            
            self.full_db_vectors = {} 
            self.active_vectors = None
            self.active_names = []
            
            # Load DB
            self._load_vector_db()
            self._prepare_search_space()
            
            # -----------------------------------------------------------
            # [MODEL LOADING]
            # -----------------------------------------------------------
            print(f"⏳ Loading YOLO Model from {CFG.MODEL_PACK}...")
            self.yolo = YOLO(CFG.MODEL_PACK, task='segment') 
            print(f"✅ YOLO Loaded!") 

            # -----------------------------------------------------------
            # [NEW] 2. LOAD OCR ENGINE
            # -----------------------------------------------------------
            print(f"⏳ Loading RapidOCR Engine...")
            self.ocr = RapidOCR()
            print(f"✅ OCR Loaded!")
            
            # State Variables
            self.latest_frame = None
            self.results = []
            self.lock = threading.Lock()
            
            # Timer Logic
            self.timer_running = False
            self.timer_start_time = 0
            self.timer_result_text = ""
            
        except Exception as e:
            print(f"❌ MODEL LOAD ERROR: {e}")
            raise e

    def _load_vector_db(self):
        if not os.path.exists(CFG.DB_PACKS_VEC):
            print("⚠️ Vector DB not found!")
            return
        with open(CFG.DB_PACKS_VEC, 'rb') as f:
            raw = pickle.load(f)
        
        for name, data in raw.items():
            norm = normalize_name(name)
            dino_list = data.get('dino', []) if isinstance(data, dict) else data
            if dino_list is not None and len(dino_list) > 0:
                self.full_db_vectors[norm] = dino_list
        print(f"✅ Loaded {len(self.full_db_vectors)} drugs from DB.")

    def _prepare_search_space(self):
        if not self.full_db_vectors: return
        vectors, names = [], []
        for norm_name, vec_list in self.full_db_vectors.items():
            for vec in vec_list:
                vectors.append(vec)
                names.append(norm_name)
        if vectors:
            self.active_vectors = np.array(vectors, dtype=np.float32).T 
            self.active_names = names

    def start_timer(self):
        self.timer_running = True
        self.timer_start_time = time.time()
        self.timer_result_text = ""
        print("⏱️ Timer Started!")

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        process_counter = 0
        while True:
            process_counter += 1
            if process_counter >= CFG.AI_FRAME_SKIP and self.latest_frame is not None:
                process_counter = 0
                self.process(self.latest_frame.copy())
            time.sleep(0.01)

    def process(self, frame: np.ndarray):
        t0 = time.perf_counter()
        
        yolo_ms = 0
        dino_ms = 0
        search_ms = 0
        ocr_ms = 0 # <--- เก็บเวลา OCR ด้วย

        if not self.rx.is_ready or self.rx.is_completed: return
        if self.active_vectors is None: return

        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # ==========================================
        # ⏱️ 1. YOLO Inference
        # ==========================================
        t_start_yolo = time.perf_counter()

        img_resized = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
        img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        results = self.yolo(img_input, conf=0.25, verbose=False)
        res = results[0]
        
        yolo_ms = (time.perf_counter() - t_start_yolo) * 1000

        if res.boxes is None or len(res.boxes) == 0:
            with self.lock: self.results = []
            return

        # 2. Crop & Scale
        sx = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE
        sy = CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
        
        crops = []
        box_coords = []
        h_orig, w_orig = frame.shape[:2]

        for box in res.boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            
            dx1, dy1 = int(x1 * sx), int(y1 * sy)
            dx2, dy2 = int(x2 * sx), int(y2 * sy)
            
            dx1, dy1 = max(0, dx1), max(0, dy1)
            dx2, dy2 = min(w_orig, dx2), min(h_orig, dy2)
            
            crop = frame[dy1:dy2, dx1:dx2]
            
            if crop.size > 0: 
                crops.append(crop)
                box_coords.append([dx1, dy1, dx2, dy2])

        temp_results = []
        if crops:
            # ==========================================
            # ⏱️ 2. DINO Feature Extraction
            # ==========================================
            t_start_dino = time.perf_counter()
            try:
                batch_dino = self.engine.extract_dino_batch(crops)
            except Exception as e:
                print(f"⚠️ DINO Error: {e}")
                return
            dino_ms = (time.perf_counter() - t_start_dino) * 1000

            # ==========================================
            # ⏱️ 3. Vector Search & Logic
            # ==========================================
            t_start_search = time.perf_counter()
            sim_matrix = np.dot(batch_dino, self.active_vectors)
            search_ms = (time.perf_counter() - t_start_search) * 1000

            # เริ่ม Loop Match ผลลัพธ์
            for i, crop_img in enumerate(crops): # <--- [IMPORTANT] เปลี่ยน _ เป็น crop_img เพื่อเอาภาพมาใช้
                
                # --- A. Vector Match ---
                best_idx = np.argmax(sim_matrix[i])
                score = sim_matrix[i][best_idx]
                matched_name = self.active_names[best_idx]
                
                # --- B. Logic เดิม ---
                display_name = "Unknown"
                is_correct = False
                
                if score > CFG.VERIFY_THRESHOLD:
                    is_correct = self.rx.verify(matched_name)
                    if is_correct:
                        display_name = self.rx.target_drugs[normalize_name(matched_name)]['original']
                        if self.timer_running:
                            elapsed = time.time() - self.timer_start_time
                            self.timer_result_text = f"{display_name} : {elapsed:.2f}s"
                            self.timer_running = False
                    else:
                        display_name = matched_name.upper()
                else:
                    display_name = f"? ({score:.2f})"

                # ==========================================
                # 📖 [NEW] C. OCR LOGGING (DEBUG MODE)
                # ==========================================
                # ทำเฉพาะตอนที่ Crop มีขนาดพอสมควร (เล็กไปอ่านไม่ออกเปลืองแรง)
                if crop_img.shape[0] > 20 and crop_img.shape[1] > 20:
                    try:
                        t_ocr_start = time.perf_counter()
                        
                        # 1. Preprocess (Gray + CLAHE)
                        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        enhanced_crop = clahe.apply(gray)
                        
                        # 2. Run OCR
                        ocr_res, _ = self.ocr(enhanced_crop)
                        
                        ocr_time = (time.perf_counter() - t_ocr_start) * 1000
                        ocr_ms += ocr_time

                        # 3. Print Log
                        if ocr_res:
                            print(f"   🔍 [OCR] Pill #{i} ({matched_name}):")
                            for item in ocr_res:
                                text, conf = item[1], item[2]
                                if conf > 0.5: # กรองพวกมั่วๆ ออก
                                    print(f"      👉 Text: '{text}' (Conf: {conf:.2f})")
                                    
                    except Exception as e:
                        print(f"   ❌ OCR Err: {e}")
                
                # --- End OCR ---

                temp_results.append({
                    'box': box_coords[i], 
                    'label': display_name, 
                    'conf': score,
                    'is_correct': is_correct
                })

        # 6. Update Shared Results
        with self.lock:
            self.results = temp_results

        # ==========================================
        # 📊 LOG SPEED
        # ==========================================
        total_ms = (time.perf_counter() - t0) * 1000
        # เพิ่ม OCR ms ใน Log จะได้รู้ว่ากินแรงแค่ไหน
        print(f"⚡ [Speed] YOLO: {yolo_ms:.0f}ms | DINO: {dino_ms:.0f}ms | OCR: {ocr_ms:.0f}ms | Total: {total_ms:.0f}ms")