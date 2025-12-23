import os

# โครงสร้างโฟลเดอร์สำหรับ Raspberry Pi (Edge Device)
rasp_structure = {
    # 1. Root Files
    "main.py": """# Orchestrator for PillTrack Edge
import time
import cv2
from src.utils.config_loader import load_config
from src.hardware.camera import CameraHandler
from src.hardware.ui_renderer import UIRenderer
from src.ai.processor import AIProcessor
from src.services.sync import SyncManager

def main():
    print("🚀 PillTrack Edge Started...")
    try:
        SyncManager().sync()
    except Exception as e:
        print(f"⚠️ Sync skipped: {e}")

    config = load_config()
    cam = CameraHandler(config)
    ai = AIProcessor(config).start()
    ui = UIRenderer(config)

    try:
        while True:
            frame = cam.get_frame()
            if frame is None: continue
            
            ai.update_frame(frame)
            results = ai.get_latest_results()
            display_frame = ui.draw(frame, results, ai.prescription_status)
            
            cv2.imshow("PillTrack", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
""",
    "config.yaml": """system:
  mode: standalone  # standalone / connected
  camera_id: 0
  
display:
  width: 1280
  height: 720

ai:
  model_path: 'models/seg_best_process.pt'
  db_path: 'database/pill_fingerprints.pkl'
  conf_threshold: 0.5

endpoints:
  his_api: 'http://localhost:8000/api'
""",
    ".env": "HIS_API_KEY=xxx\nDEVICE_ID=rasp_001\n",
    ".gitignore": "__pycache__/\n*.pyc\n.env\n.DS_Store\nlogs/\nmodels/\ndatabase/\n",
    "requirements.txt": "opencv-python\nnumpy\nonnxruntime\nultralytics\nrequests\npyyaml\npython-dotenv\n",

    # 2. Source Code (src/)
    "src/__init__.py": "",
    
    # 2.1 Hardware
    "src/hardware/__init__.py": "",
    "src/hardware/camera.py": """import cv2
class CameraHandler:
    def __init__(self, config):
        self.cap = cv2.VideoCapture(config['system'].get('camera_id', 0))
    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None
    def release(self):
        self.cap.release()
""",
    "src/hardware/ui_renderer.py": """import cv2
class UIRenderer:
    def __init__(self, config):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    def draw(self, frame, results, status):
        # Placeholder for drawing logic
        cv2.putText(frame, f"Status: {status}", (20, 50), self.font, 1, (0, 255, 0), 2)
        return frame
""",

    # 2.2 AI Logic
    "src/ai/__init__.py": "",
    "src/ai/engine.py": """class FeatureEngine:\n    def __init__(self):\n        pass\n""",
    "src/ai/processor.py": """import threading
class AIProcessor:
    def __init__(self, config):
        self.results = []
        self.prescription_status = "Ready"
    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self
    def update_frame(self, frame):
        pass
    def get_latest_results(self):
        return self.results
    def _loop(self):
        while True:
            # AI Inference Logic here
            pass
""",

    # 2.3 Core Business Logic
    "src/core/__init__.py": "",
    "src/core/prescription.py": """class PrescriptionManager:\n    def __init__(self):\n        self.target_drugs = {}\n""",

    # 2.4 Services
    "src/services/__init__.py": "",
    "src/services/his.py": """class HISConnector:\n    def fetch_data(self, hn):\n        pass\n""",
    "src/services/sync.py": """class SyncManager:\n    def sync(self):\n        print("Syncing data from Cloud...")\n""",

    # 2.5 Utils
    "src/utils/__init__.py": "",
    "src/utils/config_loader.py": """import yaml\ndef load_config():\n    with open('config.yaml') as f: return yaml.safe_load(f)\n""",

    # 3. Data Folders
    "logs/": None,
    "models/": None,
    "database/": None,
    "mock_server/": None,
    "tests/": None,
}

def create_rasp_project():
    print("🤖 Starting Raspberry Pi Project Generator...")
    
    for path, content in rasp_structure.items():
        if path.endswith("/"):
            os.makedirs(path, exist_ok=True)
            print(f"   📂 Created dir:  {path}")
        else:
            dir_name = os.path.dirname(path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"   📄 Created file: {path}")
            else:
                print(f"   ⚠️  Skipped (Exists): {path}")

    print("\n✅ Raspberry Pi Structure Created Successfully!")

if __name__ == "__main__":
    create_rasp_project()