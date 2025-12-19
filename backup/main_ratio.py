#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║ PILLTRACK: TIMER DASHBOARD EDITION                           ║
║ - Press 'T' to start timer                                   ║
║ - Time is displayed in the DASHBOARD list after [OK]         ║
║ - STRICT RGB888 Pipeline & High Visibility UI                ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import cv2
import torch
import pickle
from torchvision import transforms
from ultralytics import YOLO

# ================= ⚙️ CONFIGURATION =================
@dataclass
class Config:
    # --- PATHS ---
    MODEL_PACK: str = 'models/seg_best_process.pt'
    MODEL_PILL: str = 'models/pills_seg.pt'
    
    # 🔥 DINOv2 DATABASES
    DB_PILLS_VEC: str = 'database/db_register/db_pills_dino.pkl'
    DB_PACKS_VEC: str = 'database/db_register/db_packs_dino.pkl'
    
    # Color & SIFT Databases
    DB_PILLS_COL: str = 'database/db_register/colors_pills.pkl'
    DB_PACKS_COL: str = 'database/db_register/colors_packs.pkl'
    IMG_DB_FOLDER: str = 'database_images' 
    PRESCRIPTION_FILE: str = 'prescription.txt'

    # Display & ROI
    DISPLAY_SIZE: Tuple[int, int] = (1280, 720)
    AI_SIZE: int = 416
    
    # 🚫 EXCLUSION ZONE (Dashboard Area)
    UI_ZONE_X_START: int = 850  # ขยายพื้นที่ Dashboard ให้กว้างขึ้นสำหรับแสดงเวลา
    UI_ZONE_Y_END: int = 300 
    
    # 🎚️ TUNING THRESHOLDS
    CONF_THRESHOLD: float = 0.3
    
    # WEIGHTS
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {'vec': 0.3, 'col': 0.2, 'sift': 0.5})
    SIFT_RATIO_TEST: float = 0.7

CFG = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 SYSTEM STARTING ON: {device} (TIMER DASHBOARD MODE)")

# ================= 🛠️ HELPER: HIGH VISIBILITY TEXT =================
def draw_text_enhanced(img, text, pos, font_scale=0.6, color=(255, 255, 255), thickness=2):
    x, y = pos
    # Outline (Black)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3)
    # Text (Color)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# ================= 🧠 PRESCRIPTION STATE MANAGER =================
class PrescriptionManager:
    def __init__(self):
        self.patient_name = "WAITING..."
        self.allowed_drugs = []
        self.verified_drugs = set()
        self.load_prescription()

    def load_prescription(self):
        if not os.path.exists(CFG.PRESCRIPTION_FILE):
            return
        try:
            with open(CFG.PRESCRIPTION_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    parts = line.split('|')
                    if len(parts) >= 3:
                        self.patient_name = parts[1].strip()
                        raw_drugs = parts[2].split(',')
                        self.allowed_drugs = [d.strip().lower() for d in raw_drugs if d.strip()]
                        break
        except Exception: pass

    def is_allowed(self, db_name):
        db_clean = db_name.lower().replace('_pack', '').replace('_pill', '')
        for allowed in self.allowed_drugs:
            if allowed in db_clean or db_clean in allowed: return True
        return False

    def verify(self, name):
        clean = name.lower().replace('_pack', '').replace('_pill', '')
        for allowed in self.allowed_drugs:
            if allowed in clean or clean in allowed:
                self.verified_drugs.add(allowed)

# ================= 🎨 FEATURE ENGINE (DINOv2 + SIFT) =================
class FeatureEngine:
    def __init__(self):
        print("🦕 Loading DINOv2 (ViT-S/14)...")
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model.eval().to(device)
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((448, 448), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("✅ DINOv2 Ready")
        except Exception as e:
            print(f"❌ DINOv2 Error: {e}")
            sys.exit(1)

        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

    @torch.no_grad()
    def get_vector(self, img_rgb):
        t = self.preprocess(img_rgb).unsqueeze(0).to(device)
        output = self.model(t)
        vec = output.flatten().cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-8)

    def get_sift_features(self, img_rgb):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return des

# ================= 🤖 AI PROCESSOR =================
class AIProcessor:
    def __init__(self):
        self.engine = FeatureEngine()
        self.rx_manager = PrescriptionManager()
        self.session_db_vec = {}
        self.session_db_sift = {}
        self.load_and_filter_db()
        try:
            self.yolo_pack = YOLO(CFG.MODEL_PACK) if os.path.exists(CFG.MODEL_PACK) else YOLO('yolov8n-seg.pt')
        except: sys.exit("❌ YOLO Error")
        self.latest_frame = None
        self.results = []
        self.lock = threading.Lock()
        self.stopped = False
        self.process_time_ms = 0.0

    def load_and_filter_db(self):
        print("🔍 Loading Database...")
        def load_pkl(path):
            if os.path.exists(path):
                with open(path, 'rb') as f: return pickle.load(f)
            return {}
        all_vecs = {**load_pkl(CFG.DB_PILLS_VEC), **load_pkl(CFG.DB_PACKS_VEC)}
        count = 0
        for name, vecs in all_vecs.items():
            if self.rx_manager.is_allowed(name):
                for v in vecs:
                    if len(v) != 384: continue
                    self.session_db_vec[f"{name}_{count}"] = (name, np.array(v))
                    count += 1
        if os.path.exists(CFG.IMG_DB_FOLDER):
            for drug_name in os.listdir(CFG.IMG_DB_FOLDER):
                if not self.rx_manager.is_allowed(drug_name): continue
                drug_path = os.path.join(CFG.IMG_DB_FOLDER, drug_name)
                if os.path.isdir(drug_path):
                    descriptors_list = []
                    for img_file in sorted(os.listdir(drug_path))[:3]:
                        if img_file.lower().endswith(('jpg', 'png', 'jpeg')):
                            img_bgr = cv2.imread(os.path.join(drug_path, img_file))
                            if img_bgr is not None:
                                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                                des = self.engine.get_sift_features(img_rgb)
                                if des is not None: descriptors_list.append(des)
                    if descriptors_list:
                        self.session_db_sift[drug_name] = descriptors_list

    def compute_sift_score(self, query_des, target_name):
        if query_des is None or target_name not in self.session_db_sift: return 0.0
        max_matches = 0
        for ref_des in self.session_db_sift[target_name]:
            try:
                matches = self.engine.bf.knnMatch(query_des, ref_des, k=2)
                good = [m for m, n in matches if m.distance < CFG.SIFT_RATIO_TEST * n.distance]
                max_matches = max(max_matches, len(good))
            except: pass
        return min(max_matches / 15.0, 1.0)

    def match(self, vec, img_crop):
        candidates = []
        if not self.session_db_vec: return []
        query_sift_des = self.engine.get_sift_features(img_crop)
        for key, (real_name, db_v) in self.session_db_vec.items():
            vec_score = np.dot(vec, db_v)
            col_score = 0.5 
            sift_score = self.compute_sift_score(query_sift_des, real_name)
            final_score = (vec_score * CFG.WEIGHTS['vec']) + (col_score * CFG.WEIGHTS['col']) + (sift_score * CFG.WEIGHTS['sift'])
            candidates.append((real_name, final_score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        unique = []
        seen = set()
        for n, fs in candidates:
            if n not in seen:
                unique.append((n, fs))
                seen.add(n)
            if len(unique) >= 3: break
        return unique

    def process_frame(self, frame_rgb):
        t_start = time.time()
        img_ai = cv2.resize(frame_rgb, (CFG.AI_SIZE, CFG.AI_SIZE))
        results = self.yolo_pack(img_ai, verbose=False, conf=0.75, imgsz=CFG.AI_SIZE, task='segment')
        detections = []
        res = results[0]
        if res.masks is None:
            with self.lock: self.results = []
            self.process_time_ms = (time.time() - t_start) * 1000
            return
        for box, mask in zip(res.boxes, res.masks):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            scale_x = CFG.DISPLAY_SIZE[0] / CFG.AI_SIZE
            scale_y = CFG.DISPLAY_SIZE[1] / CFG.AI_SIZE
            rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
            rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)
            cx, cy = (rx1+rx2)//2, (ry1+ry2)//2
            if cx > CFG.UI_ZONE_X_START and cy < CFG.UI_ZONE_Y_END: continue
            contour = mask.xyn[0]
            contour[:, 0] *= CFG.DISPLAY_SIZE[0]
            contour[:, 1] *= CFG.DISPLAY_SIZE[1]
            contour = contour.astype(np.int32)
            crop = frame_rgb[ry1:ry2, rx1:rx2]
            if crop.size == 0: continue
            vec = self.engine.get_vector(crop)
            candidates = self.match(vec, crop)
            label = "Unknown"
            score = 0.0
            if candidates:
                top_name, top_score = candidates[0]
                if top_score > CFG.CONF_THRESHOLD:
                    label = top_name
                    self.rx_manager.verify(label)
                score = top_score
            detections.append({'box': (rx1, ry1, rx2, ry2), 'contour': contour, 'label': label, 'score': score})
        with self.lock: self.results = detections
        self.process_time_ms = (time.time() - t_start) * 1000

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self
    def _run(self):
        while not self.stopped:
            with self.lock: frame = self.latest_frame
            if frame is not None:
                try: self.process_frame(frame)
                except Exception as e: print(f"Err: {e}")
            time.sleep(0.01)

# ================= 📷 CAMERA =================
class Camera:
    def __init__(self):
        self.cap = None
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2()
            cfg = self.picam.create_preview_configuration(main={"size": CFG.DISPLAY_SIZE, "format": "RGB888"})
            self.picam.configure(cfg)
            self.picam.start()
            self.use_pi = True
            print("📷 PiCamera2: NATIVE RGB888 MODE")
        except:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, CFG.DISPLAY_SIZE[0])
            self.cap.set(4, CFG.DISPLAY_SIZE[1])
            self.use_pi = False
            print("📷 USB Camera: FORCED RGB CONVERSION ACTIVE")

    def get(self):
        if self.use_pi: return self.picam.capture_array()
        else:
            ret, frame_bgr = self.cap.read()
            if ret: return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return None
    def stop(self):
        if self.use_pi: self.picam.stop()
        else: self.cap.release()

# ================= 🖥️ UI RENDERER (RGB + TIMER) =================
def draw_ui(frame_rgb, results, rx_manager, fps, ai_time_ms, timer_state):
    h, w = frame_rgb.shape[:2]
    
    is_timer_active = timer_state['active']
    timer_start_t = timer_state['start_time']
    found_times = timer_state['found_times']
    
    current_elapsed = 0.0
    if is_timer_active:
        current_elapsed = time.time() - timer_start_t

    # --- 1. Dashboard Background ---
    db_x, db_y = CFG.UI_ZONE_X_START, 10
    db_w, db_h = w - db_x - 10, CFG.UI_ZONE_Y_END
    sub = frame_rgb[db_y:db_y+db_h, db_x:db_x+db_w]
    black_rect = np.zeros(sub.shape, dtype=np.uint8) 
    cv2.addWeighted(sub, 0.5, black_rect, 0.5, 0, sub)
    cv2.rectangle(frame_rgb, (db_x, db_y), (db_x+db_w, db_y+db_h), (0, 255, 255), 2)

    # --- 2. Timer Status ---
    draw_text_enhanced(frame_rgb, "TIMER (Press 'T')", (db_x+10, db_y+30), 0.6, (0, 255, 255), 2)
    if is_timer_active:
        draw_text_enhanced(frame_rgb, f"⏱️ {current_elapsed:.2f} s", (db_x+10, db_y+60), 0.7, (0, 255, 0), 2)
    else:
        draw_text_enhanced(frame_rgb, "⏱️ STOPPED", (db_x+10, db_y+60), 0.7, (150, 150, 150), 2)

    # --- 3. Performance Stats ---
    draw_text_enhanced(frame_rgb, f"FPS: {fps:.1f} | AI: {ai_time_ms:.0f}ms", (db_x+10, db_y+90), 0.5, (200, 200, 200), 1)
    cv2.line(frame_rgb, (db_x+5, db_y+100), (db_x+db_w-5, db_y+100), (200, 200, 200), 1)

    # --- 4. Prescription List (แก้ไขตรงนี้ให้แสดงเวลาหลัง OK) ---
    y_off = 130
    for drug in rx_manager.allowed_drugs:
        is_verified = drug in rx_manager.verified_drugs
        
        status_suffix = ""
        item_color = (180, 180, 180) # Grey text for unverified
        
        if is_verified:
            item_color = (0, 255, 0) # Green text for verified
            status_suffix = " [OK]"
            
            # 🔥 เช็คว่ามีเวลาที่บันทึกไว้ไหม ถ้ามีให้เอามาต่อท้าย OK
            if drug in found_times:
                t_found = found_times[drug]
                status_suffix += f" ({t_found:.2f}s)" # e.g. " [OK] (1.23s)"
        else:
            status_suffix = " [...]"

        # แสดงผล: "- Paracetamol [OK] (1.23s)"
        draw_text_enhanced(frame_rgb, f"- {drug}{status_suffix}", (db_x+10, db_y+y_off), 0.6, item_color, 2)
        y_off += 30

    # --- 5. Draw Detections ---
    for det in results:
        contour = det['contour']
        label = det['label']
        score = det['score']
        
        color = (0, 255, 0) if label != "Unknown" else (255, 0, 0)
        cv2.drawContours(frame_rgb, [contour], -1, color, 3)
        
        top_point = tuple(contour[contour[:, 1].argmin()])
        tx, ty = top_point
        
        # ป้ายชื่อบนยา แสดงแค่ชื่อกับ % (เอาเวลาออกให้ดูสะอาดตา ตามที่เข้าใจ)
        label_str = f"{label} ({score:.0%})"
        
        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame_rgb, (tx, ty - 30), (tx + tw + 10, ty), (0, 0, 0), -1)
        cv2.putText(frame_rgb, label_str, (tx + 5, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ================= 🚀 MAIN LOOP =================
if __name__ == "__main__":
    cam = Camera()
    ai = AIProcessor().start()
    
    print("✨ WAITING FOR FRAMES (Press 'T' to start timer)...")
    while cam.get() is None: time.sleep(0.1)
    
    cv2.namedWindow("PillTrack Timer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PillTrack Timer", *CFG.DISPLAY_SIZE)
    
    prev_time = time.time()
    fps_avg = 0.0
    
    timer_state = {
        'active': False,
        'start_time': 0.0,
        'found_times': {} 
    }
    
    try:
        while True:
            key = cv2.waitKey(1)
            if key == ord('q'): break
            elif key == ord('t') or key == ord('T'):
                timer_state['active'] = True
                timer_state['start_time'] = time.time()
                timer_state['found_times'] = {} # Reset times
                print("⏱️ TIMER STARTED / RESET")

            frame_rgb = cam.get()
            if frame_rgb is None: continue
            
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            fps_avg = (0.9 * fps_avg) + (0.1 * fps)
            
            if timer_state['active']:
                current_duration = curr_time - timer_state['start_time']
                with ai.lock: current_results = ai.results
                for det in current_results:
                    name = det['label']
                    if name != "Unknown" and name not in timer_state['found_times']:
                        # บันทึกเวลาเมื่อเจอครั้งแรก
                        timer_state['found_times'][name] = current_duration
                        print(f"🎯 FOUND: {name} in {current_duration:.2f}s")

            ai.latest_frame = frame_rgb.copy()
            draw_ui(frame_rgb, ai.results, ai.rx_manager, fps_avg, ai.process_time_ms, timer_state)
            
            cv2.imshow("PillTrack Timer", frame_rgb)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cam.stop()
        ai.stopped = True
        cv2.destroyAllWindows()