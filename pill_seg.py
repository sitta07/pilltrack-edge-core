#!/usr/bin/env python3
"""
PILLTRACK – PURE RGB SEGMENTATION (FIXED PLOT) 🌈
✔ Fix 'unexpected argument alpha' error by removing it.
✔ Keeps entire pipeline in RGB.
"""

import time
import yaml
import cv2
import torch
import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO

# ================= ⚙️ CONFIG =================
try:
    with open("config.yaml", "r") as f:
        yaml_cfg = yaml.safe_load(f)
except FileNotFoundError:
    yaml_cfg = {}

@dataclass
class Config:
    # ⚠️ Check model path (must be a segmentation model e.g., yolov8n-seg.pt)
    MODEL_PATH: str = yaml_cfg.get('artifacts', {}).get('model', 'yolov8n-seg.pt') 
    
    DISPLAY_WIDTH: int = yaml_cfg.get('display', {}).get('width', 1280)
    DISPLAY_HEIGHT: int = yaml_cfg.get('display', {}).get('height', 720)
    
    CONF_THRESHOLD: float = 0.5
    AI_SIZE: int = 640 

CFG = Config()

# ================= 📷 CAMERA HANDLER (FORCE RGB) =================
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
            print("📷 Camera: Using Picamera2 (Force RGB Mode)")
        except ImportError:
            print("⚠️ Picamera2 not found. Switching to OpenCV.")
        except Exception as e:
            print(f"⚠️ Picamera2 failed: {e}. Switching to OpenCV.")

        if not self.use_picamera:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_rgb_frame(self):
        """Returns a clean 3-Channel RGB Frame."""
        if self.use_picamera:
            raw = self.picam.capture_array()
            # BGRA -> RGB
            return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        else:
            ret, frame = self.cap.read()
            if not ret: return None
            # BGR -> RGB
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        if self.use_picamera: self.picam.stop()
        elif self.cap: self.cap.release()

# ================= 🤖 AI SEGMENTOR =================
class Segmentor:
    def __init__(self):
        print(f"⏳ Loading YOLO Segmentation: {CFG.MODEL_PATH}")
        self.model = YOLO(CFG.MODEL_PATH)
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()

    def process(self, rgb_frame):
        # 1. Inference on RGB Image
        results = self.model(rgb_frame, 
                             conf=CFG.CONF_THRESHOLD, 
                             imgsz=CFG.AI_SIZE, 
                             retina_masks=True, 
                             verbose=False)
        res = results[0]

        # 2. Draw Segmentation Overlay (RGB)
        # ⚠️ FIXED: Removed 'alpha=0.4' for compatibility with older ultralytics versions
        annotated_rgb = res.plot(img=rgb_frame.copy()) 

        # 3. FPS Calculation
        self.frame_count += 1
        if time.time() - self.prev_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.prev_time = time.time()

        # Draw FPS (Green = RGB correct)
        cv2.putText(annotated_rgb, f"FPS: {self.fps}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated_rgb

# ================= 🚀 MAIN LOOP =================
def main():
    try: 
        camera = CameraHandler(width=CFG.DISPLAY_WIDTH, height=CFG.DISPLAY_HEIGHT)
        ai = Segmentor()
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return
    
    print("🚀 PillTrack Segmentation Started (RGB Mode)")
    print("⌨️  Press [Q] to Quit")

    while True:
        # 1. Get RGB Frame
        rgb_frame = camera.get_rgb_frame()
        if rgb_frame is None:
            time.sleep(0.01)
            continue
        
        # 2. Process (RGB -> RGB)
        final_rgb = ai.process(rgb_frame)
        
        # 3. Display (Convert to BGR only for imshow correct colors)
        display_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("PillTrack Segment", display_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("👋 Exiting...")

if __name__ == "__main__":
    main()