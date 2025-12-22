#!/usr/bin/env python3
"""
PILLTRACK – PURE RGB SEGMENTATION 🌈
✔ Fix 4-channel error (BGRA -> RGB)
✔ Pipeline operates in RGB for AI compatibility
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
            # Picamera returns 4 channels (XRGB/BGRA) usually
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
        """
        Returns a clean 3-Channel RGB Frame.
        Solves the 'expected 3 channels, got 4' error.
        """
        if self.use_picamera:
            # Raw is 4 channels (BGRA/XRGB)
            raw = self.picam.capture_array()
            # Convert 4 channels -> 3 channels RGB
            return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        else:
            ret, frame = self.cap.read()
            if not ret: return None
            # OpenCV Raw is BGR -> Convert to RGB
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
        # 1. Inference on RGB Image (3 Channels)
        # retina_masks=True for high-quality masks
        results = self.model(rgb_frame, 
                             conf=CFG.CONF_THRESHOLD, 
                             imgsz=CFG.AI_SIZE, 
                             retina_masks=True, 
                             verbose=False)
        res = results[0]

        # 2. Draw Segmentation Overlay directly on the RGB image
        annotated_rgb = res.plot(img=rgb_frame.copy(), alpha=0.4) 

        # 3. FPS Calculation
        self.frame_count += 1
        if time.time() - self.prev_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.prev_time = time.time()

        # Draw FPS on RGB image
        cv2.putText(annotated_rgb, f"FPS: {self.fps}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) # Green text

        return annotated_rgb

# ================= 🚀 MAIN LOOP =================
def main():
    try: 
        # Initialize Camera
        camera = CameraHandler(width=CFG.DISPLAY_WIDTH, height=CFG.DISPLAY_HEIGHT)
        # Initialize AI
        ai = Segmentor()
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        return
    
    print("🚀 PillTrack Segmentation Started (RGB Mode)")
    print("⌨️  Press [Q] to Quit")

    while True:
        # 1. Get Clean RGB Frame (3 Channels)
        rgb_frame = camera.get_rgb_frame()
        
        if rgb_frame is None:
            time.sleep(0.01)
            continue
        
        # 2. Process with AI (RGB in -> RGB out)
        final_rgb = ai.process(rgb_frame)
        
        # 3. Display
        # Note: cv2.imshow expects BGR format to display colors correctly on screen.
        # We convert RGB -> BGR only for this specific line.
        display_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("PillTrack Segment", display_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("👋 Exiting...")

if __name__ == "__main__":
    main()