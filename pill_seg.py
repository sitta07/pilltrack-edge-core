#!/usr/bin/env python3
"""
PILLTRACK – PURE YOLO SEGMENTATION 🎨
✔ ตัดทุกอย่าง เหลือแค่กล้อง + AI Segmentation
✔ วาด Mask สีๆ ทับวัตถุที่เจอ
"""

import os
import time
import yaml
import numpy as np
import cv2
import torch
from dataclasses import dataclass
from ultralytics import YOLO

# ================= ⚙️ CONFIG =================
# โหลด config เอาแค่ resolution กับโมเดลพอ
try:
    with open("config.yaml", "r") as f:
        yaml_cfg = yaml.safe_load(f)
except FileNotFoundError:
    yaml_cfg = {} # Fallback ถ้าไม่มีไฟล์

@dataclass
class Config:
    # ⚠️ อย่าลืมแก้ Path Model ให้เป็นตัวที่เทรน Seg มานะ (เช่น best-seg.pt)
    MODEL_PATH: str = yaml_cfg.get('artifacts', {}).get('model', 'yolov12-seg(last).pt') 
    
    DISPLAY_WIDTH: int = yaml_cfg.get('display', {}).get('width', 1280)
    DISPLAY_HEIGHT: int = yaml_cfg.get('display', {}).get('height', 720)
    
    CONF_THRESHOLD: float = 0.5
    AI_SIZE: int = 640 # ปกติ Seg ใช้ 640 จะแม่นกว่า 416

CFG = Config()

# ================= 📷 CAMERA HANDLER =================
# (ใช้ตัวเดิม เพราะเขียนไว้ดีแล้ว รองรับทั้ง Picamera/Webcam)
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
        except ImportError:
            print("⚠️ Picamera2 not found. Switching to OpenCV.")
            self.use_picamera = False
        except Exception as e:
            print(f"⚠️ Picamera2 failed: {e}. Falling back...")
            self.use_picamera = False

        if not self.use_picamera:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_frame(self):
        if self.use_picamera:
            return self.picam.capture_array()
        else:
            ret, frame = self.cap.read()
            if not ret: return None
            # OpenCV อ่านเป็น BGR ต้องแปลงเป็น RGB/RGBA ให้ตรง format
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    def release(self):
        if self.use_picamera: self.picam.stop()
        elif self.cap: self.cap.release()

# ================= 🤖 AI & VISUALIZATION =================
class Segmentor:
    def __init__(self):
        print(f"⏳ Loading YOLO Segmentation Model: {CFG.MODEL_PATH}")
        self.model = YOLO(CFG.MODEL_PATH)
        self.frame_count = 0
        self.fps = 0
        self.prev_time = time.time()

    def process_and_draw(self, frame):
        # 1. Inference (เปิด retina_masks=True เพื่อความคม)
        results = self.model(frame, conf=CFG.CONF_THRESHOLD, imgsz=CFG.AI_SIZE, retina_masks=True, verbose=False)
        res = results[0]

        # 2. Plotting (ใช้ฟังก์ชัน plot ของ ultralytics เลย ง่ายและสวยสุด)
        # มันจะวาด Mask + Box + Label ให้เองแบบโปร่งใส
        annotated_frame = res.plot(img=frame.copy(), alpha=0.4) 

        # คำนวณ FPS เล่นๆ
        self.frame_count += 1
        if time.time() - self.prev_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.prev_time = time.time()

        # แปะ FPS มุมซ้ายบน
        cv2.putText(annotated_frame, f"FPS: {self.fps}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return annotated_frame

# ================= 🚀 MAIN =================
def main():
    try: 
        camera = CameraHandler(width=CFG.DISPLAY_WIDTH, height=CFG.DISPLAY_HEIGHT)
    except Exception as e:
        print(f"❌ Camera Error: {e}")
        return

    ai = Segmentor()
    
    print("🚀 Segmentation Mode Started!")
    print("⌨️  Press [Q] to Quit")

    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        # ส่งเข้า AI แล้วรับภาพที่วาดแล้วกลับมา
        final_frame = ai.process_and_draw(frame)
        
        # แสดงผล (Convert กลับเป็น BGR เพื่อให้สีถูกต้องบน opencv window)
        cv2.imshow("PillTrack: YOLO Segmentation", cv2.cvtColor(final_frame, cv2.COLOR_RGBA2BGR))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("👋 Bye Bye!")

if __name__ == "__main__":
    main()