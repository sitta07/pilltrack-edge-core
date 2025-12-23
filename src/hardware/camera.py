import cv2
from src.utils.config import CFG

class CameraHandler:
    def __init__(self):
        self.width = CFG.DISPLAY_SIZE[0]
        self.height = CFG.DISPLAY_SIZE[1]
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
            print("⚠️ Picamera2 not found. Falling back to OpenCV.")
            self._init_opencv()
        except Exception as e:
            print(f"⚠️ Camera Error: {e}. Falling back to OpenCV.")
            self._init_opencv()

    def _init_opencv(self):
        self.use_picamera = False
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_frame(self):
        if self.use_picamera:
            return self.picam.capture_array()
        else:
            if not self.cap.isOpened(): return None
            ret, frame = self.cap.read()
            if not ret: return None
            # OpenCV is BGR -> Convert to RGB/RGBA
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    def release(self):
        if self.use_picamera and self.picam:
            self.picam.stop()
        elif self.cap:
            self.cap.release()