import threading
import time
import pickle
import os
import cv2
import numpy as np
from ultralytics import YOLO
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
            
            # --- [MODIFIED] Loading Model Logic ---
            print(f"⏳ Loading Detection Model from {CFG.MODEL_PACK}...")
            
            # task='detect' สำคัญมากสำหรับ ONNX เพื่อให้ Library รู้ว่าจะเตรียม Output แบบไหน
            self.yolo = YOLO(CFG.MODEL_PACK, task='detect') 
            
            # Warmup (Optional): รันภาพเปล่า 1 ครั้ง เพื่อให้ ONNX Runtime โหลด Provider รอไว้เลย
            # dummy_img = np.zeros((CFG.AI_SIZE, CFG.AI_SIZE, 3), dtype=np.uint8)
            # self.yolo(dummy_img, verbose=False)
            
            print(f"✅ Model Loaded successfully! (Source: {CFG.MODEL_PACK})") 
            
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
            # 0. Safety Checks
            if not self.rx.is_ready or self.rx.is_completed: return
            if self.active_vectors is None: return

            # ---------------------------------------------------------
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]  # ตัดช่อง Alpha ทิ้งไป
            # ---------------------------------------------------------

            # 1. YOLO Detect
            # Resize ลงมาตาม config
            img_resized = cv2.resize(frame, (CFG.AI_SIZE, CFG.AI_SIZE))
            
            # [MODIFIED] Inference
            # Ultralytics จะจัดการ ONNX backend ให้เอง ได้ผลลัพธ์เป็น Results object เหมือนเดิมเป๊ะ
            res = self.yolo(img_resized, conf=CFG.CONF_THRESHOLD, verbose=False)[0]

            if res.boxes is None or len(res.boxes) == 0:
                with self.lock: self.results = []
                return

            # 2. Crop & Scale Coordinates
            sx = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE
            sy = CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
            
            crops = []
            box_coords = []
            
            h_orig, w_orig = frame.shape[:2]

            for box in res.boxes:
                # .xyxy[0] ใช้งานได้เหมือนเดิมไม่ว่าจะเป็น PT หรือ ONNX
                x1, y1, x2, y2 = box.xyxy[0].cpu().int().tolist() # เพิ่ม .cpu() กันเหนียวเผื่อรัน GPU
                
                # Scale coordinates
                dx1, dy1 = int(x1 * sx), int(y1 * sy)
                dx2, dy2 = int(x2 * sx), int(y2 * sy)
                
                # Clamp coordinates
                dx1, dy1 = max(0, dx1), max(0, dy1)
                dx2, dy2 = min(w_orig, dx2), min(h_orig, dy2)
                
                crop = frame[dy1:dy2, dx1:dx2]
                
                if crop.size > 0: 
                    crops.append(crop)
                    box_coords.append([dx1, dy1, dx2, dy2])

            temp_results = []
            if crops:
                # 3. DINO Feature Extraction
                try:
                    batch_dino = self.engine.extract_dino_batch(crops)
                except Exception as e:
                    print(f"⚠️ DINO Error: {e}")
                    return

                # 4. Global Search
                sim_matrix = np.dot(batch_dino, self.active_vectors)
                
                for i, _ in enumerate(crops):
                    best_idx = np.argmax(sim_matrix[i])
                    score = sim_matrix[i][best_idx]
                    matched_name = self.active_names[best_idx]
                    
                    display_name = "Unknown"
                    is_correct = False
                    
                    # 5. Verification Logic
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

                    temp_results.append({
                        'box': box_coords[i], 
                        'label': display_name, 
                        'conf': score,
                        'is_correct': is_correct
                    })

            # 6. Update Shared Results
            with self.lock:
                self.results = temp_results