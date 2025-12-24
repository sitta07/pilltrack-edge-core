import cv2
import time
from src.utils.config import CFG

class CameraHandler:
    def __init__(self):
        # 1. ดึงค่า Config ขนาดภาพ
        self.width = CFG.DISPLAY_SIZE[0]
        self.height = CFG.DISPLAY_SIZE[1]
        self.cap = None
        
        # 2. เริ่มต้นกล้อง USB (ตัด Picamera ทิ้งไปเลย)
        print("📷 Initializing USB Camera...")
        self._init_opencv()

    def _init_opencv(self):
        """เปิดกล้อง USB ด้วย OpenCV"""
        try:
            # เลข 0 คือกล้องตัวแรก (ถ้าเสียบหลายตัวอาจจะเป็น 1, 2)
            self.cap = cv2.VideoCapture(0)
            
            # ตั้งค่าความละเอียด
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # ตรวจสอบว่าเปิดติดไหม
            if not self.cap.isOpened():
                raise Exception("Could not open video device (Index 0)")
                
            print(f"✅ Camera Started: {self.width}x{self.height}")
            
        except Exception as e:
            print(f"❌ Camera Error: {e}")
            self.cap = None

    def get_frame(self):
        """อ่านภาพ 1 เฟรม"""
        if self.cap is None or not self.cap.isOpened():
            print("⚠️ Camera is not opened, trying to reconnect...")
            self._init_opencv()
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            return None

        # ✅ แก้ตรงนี้: แปลงจาก BGR (ค่าเดิม) -> RGB (เพื่อให้ AI เข้าใจถูก)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        """คืนค่ากล้องเมื่อปิดโปรแกรม"""
        if self.cap:
            self.cap.release()
            print("📷 Camera Released")