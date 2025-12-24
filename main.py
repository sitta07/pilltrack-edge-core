#!/usr/bin/env python3
import time
import cv2
import os
import json
import threading
from collections import deque
from dotenv import load_dotenv
from flask import Flask, Response, jsonify

# Import Modular Components
from src.hardware.camera import CameraHandler
from src.hardware.ui_renderer import UIRenderer
from src.ai.processor import AIProcessor
from src.services.his import HISConnector
from src.services.sync import SyncManager
from src.utils.config import CFG

load_dotenv()

# ==========================================
# 🌐 1. Setup Web Server (Flask)
# ==========================================
app = Flask(__name__)

# ตัวแปรกลางสำหรับส่งภาพจาก AI ไปหน้าเว็บ
global_frame = None
lock = threading.Lock()

# ตัวแปรควบคุม
command_queue = deque()
hn_queue = deque(["HN123", "HN456"])
current_hn = "WAITING"

# ==========================================
# 🧠 2. Background AI Loop
# ==========================================
def run_ai_system():
    global global_frame, current_hn, hn_queue

    print("🚀 Initializing PillTrack Edge (Server Mode)...")
    
    # --- Sync Process ---
    try:
        print("🔄 Connecting to S3 system...")
        SyncManager().sync()
    except Exception as e:
        print(f"❌ S3 Error: {e}")

    # --- Hardware Setup ---
    try:
        camera = CameraHandler()
        ai = AIProcessor().start()
        ui = UIRenderer()
        his = HISConnector()
    except Exception as e:
        print(f"❌ Critical Init Error: {e}")
        return

    # --- Load Mock Data ---
    MOCK_DB_PATH = "mock_server/prescriptions.json"
    if os.path.exists(MOCK_DB_PATH):
        try:
            with open(MOCK_DB_PATH, 'r') as f:
                hn_queue = deque(list(json.load(f).keys()))
        except: pass
    
    print("✅ System Ready. Web Dashboard available.")

    # --- Main Loop ---
    while True:
        # 1. เช็กคำสั่งจากหน้าเว็บ
        if command_queue:
            cmd = command_queue.popleft()
            if cmd == 'timer':
                ai.start_timer()
                print("⏳ Timer Started via Web")
            elif cmd == 'next':
                if hn_queue:
                    hn_queue.rotate(-1)
                    current_hn = hn_queue[0]
                    print(f"⏩ Next Patient via Web: {current_hn}")
                    ai.rx.reset()
                    ai.timer_result_text = ""
                    data = his.fetch_prescription(current_hn)
                    if data: ai.rx.update_from_his(data)

        # 2. อ่านภาพ + AI
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        ai.latest_frame = frame
        display_frame = frame.copy()

        # 3. วาด UI
        if ai.rx.is_ready:
            display_frame = ui.draw(display_frame, ai)
            
            # Auto Reset
            if ai.rx.is_completed and (time.time() - ai.rx.complete_timestamp > 3.0):
                ai.rx.reset()
                ai.timer_result_text = ""

        # 4. ส่งภาพออกไปที่ Global Variable (ต้องแปลงกลับเป็น BGR เพื่อส่งออก Web Stream)
        with lock:
            # OpenCV ใช้ RGB (จากการแก้ของเรา) -> แต่ Web Stream ต้องการ BGR/JPEG
            # (จริงๆ JPEG รับได้หมด แต่ถ้าสีเพี้ยนให้แก้บรรทัดนี้ครับ)
            out_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            global_frame = cv2.imencode('.jpg', out_frame)[1].tobytes()

    camera.release()

# ==========================================
# 📡 3. Web Routes
# ==========================================
@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>PillTrack Edge</title>
            <style>
                body { font-family: sans-serif; text-align: center; background: #1a1a1a; color: white; margin: 0; padding: 20px; }
                h1 { margin-bottom: 10px; }
                .container { display: flex; flex-direction: column; align-items: center; }
                .video-box { border: 3px solid #444; border-radius: 8px; overflow: hidden; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
                .controls { margin-top: 20px; display: flex; gap: 20px; }
                button { padding: 15px 30px; font-size: 18px; border: none; border-radius: 50px; cursor: pointer; transition: 0.2s; font-weight: bold; }
                .btn-next { background: #007bff; color: white; }
                .btn-next:hover { background: #0056b3; }
                .btn-timer { background: #28a745; color: white; }
                .btn-timer:hover { background: #1e7e34; }
                .btn-next:active, .btn-timer:active { transform: scale(0.95); }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>💊 PillTrack Edge Monitor</h1>
                <div class="video-box">
                    <img src="/video_feed" style="width: 100%; max-width: 960px; height: auto;">
                </div>
                <div class="controls">
                    <button class="btn-next" onclick="fetch('/cmd/next')">⏩ Next Patient</button>
                    <button class="btn-timer" onclick="fetch('/cmd/timer')">⏱️ Start Timer</button>
                </div>
            </div>
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if global_frame is None:
                    time.sleep(0.01)
                    continue
                frame_data = global_frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cmd/<action>')
def command(action):
    command_queue.append(action)
    return jsonify({"status": "ok", "action": action})

if __name__ == "__main__":
    # Start AI Thread
    t = threading.Thread(target=run_ai_system)
    t.daemon = True
    t.start()

    # Start Web Server
    print(f"🌍 Dashboard live at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)