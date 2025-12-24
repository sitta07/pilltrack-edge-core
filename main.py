#!/usr/bin/env python3
import time
import cv2
import os
import json
import threading
from collections import deque
from dotenv import load_dotenv
from flask import Flask, Response, render_template_string, jsonify

# Import Modular Components
from src.hardware.camera import CameraHandler
from src.hardware.ui_renderer import UIRenderer
from src.ai.processor import AIProcessor
from src.services.his import HISConnector
from src.services.sync import SyncManager

load_dotenv()

# ==========================================
# 🌐 1. Setup Web Server (Flask)
# ==========================================
app = Flask(__name__)

# ตัวแปรกลางสำหรับส่งภาพจาก AI ไปหน้าเว็บ
global_frame = None
lock = threading.Lock()

# ตัวแปรควบคุม (แทนการกดปุ่ม)
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

    # --- Main Loop (เหมือนเดิมแต่ตัด GUI ออก) ---
    while True:
        # 1. เช็กคำสั่งจากหน้าเว็บ (แทน cv2.waitKey)
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

        # 4. ส่งภาพออกไปที่ Global Variable (เพื่อให้เว็บเอาไปโชว์)
        with lock:
            # ต้องแปลงเป็น JPEG รอไว้เลยเพื่อความเร็ว
            global_frame = cv2.imencode('.jpg', cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))[1].tobytes()

    camera.release()

# ==========================================
# 📡 3. Web Routes (ส่วนติดต่อผู้ใช้)
# ==========================================
@app.route('/')
def index():
    # หน้า Dashboard ง่ายๆ พร้อมปุ่มกด
    return """
    <html>
        <head>
            <title>PillTrack Edge Dashboard</title>
            <style>
                body { font-family: sans-serif; text-align: center; background: #222; color: white; }
                .btn { padding: 15px 30px; font-size: 20px; margin: 10px; cursor: pointer; border: none; border-radius: 5px; }
                .btn-green { background: #28a745; color: white; }
                .btn-blue { background: #007bff; color: white; }
                img { border: 2px solid #555; max-width: 100%; }
            </style>
        </head>
        <body>
            <h1>💊 PillTrack Edge Monitor</h1>
            <img src="/video_feed" width="800">
            <br><br>
            <button class="btn btn-blue" onclick="sendCommand('next')">⏩ Next Patient</button>
            <button class="btn btn-green" onclick="sendCommand('timer')">⏱️ Start Timer</button>

            <script>
                function sendCommand(cmd) {
                    fetch('/cmd/' + cmd);
                }
            </script>
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if global_frame is None:
                    continue
                frame_data = global_frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cmd/<action>')
def command(action):
    command_queue.append(action)
    return jsonify({"status": "ok", "action": action})

# ==========================================
# 🚀 4. Main Entry Point
# ==========================================
if __name__ == "__main__":
    # รัน AI ใน Thread แยก (ทำงานเบื้องหลัง)
    t = threading.Thread(target=run_ai_system)
    t.daemon = True
    t.start()

    # รัน Web Server (Main Thread)
    # host='0.0.0.0' เพื่อให้ Mac เครื่องอื่นมองเห็นได้
    print(f"🌍 Dashboard live at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)