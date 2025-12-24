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
# ⚙️ Shared Resources (กองกลาง)
# ==========================================
app = Flask(__name__)

class SharedState:
    def __init__(self):
        self.frame = None           # ภาพสดล่าสุดจากกล้อง
        self.processed_frame = None # ภาพที่วาด UI เสร็จแล้ว (พร้อมส่งขึ้นเว็บ)
        self.lock = threading.Lock()
        self.running = True
        
        # Command Queue
        self.command_queue = deque()
        
        # Data
        self.hn_queue = deque(["HN123", "HN456"])
        self.current_hn = "WAITING"

state = SharedState()

# ==========================================
# 📸 Thread 1: Camera Worker (ทำงานเร็วสุด)
# ==========================================
def camera_worker():
    print("📷 Camera Thread Started...")
    camera = CameraHandler()
    
    while state.running:
        raw_frame = camera.get_frame()
        if raw_frame is not None:
            with state.lock:
                state.frame = raw_frame.copy()
        
        # พักนิดเดียวเพื่อให้ CPU หายใจ (ประมาณ 60 FPS cap)
        time.sleep(0.015) 
    
    camera.release()
    print("📷 Camera Thread Stopped.")

# ==========================================
# 🧠 Thread 2: AI Worker (ทำงานหนักสุด)
# ==========================================
def ai_worker():
    print("🧠 AI Thread Started...")
    
    # Init System
    try:
        SyncManager().sync()
        ai = AIProcessor().start() # สตาร์ทโหลด model
        his = HISConnector()
        ui = UIRenderer()
        
        # Load Mock Data
        MOCK_DB_PATH = "mock_server/prescriptions.json"
        if os.path.exists(MOCK_DB_PATH):
            try:
                with open(MOCK_DB_PATH, 'r') as f:
                    state.hn_queue = deque(list(json.load(f).keys()))
            except: pass
            
    except Exception as e:
        print(f"❌ AI Init Error: {e}")
        return

    while state.running:
        # 1. ดึงภาพล่าสุดจากกองกลาง (ถ้าไม่มีก็วนรอ)
        input_frame = None
        with state.lock:
            if state.frame is not None:
                input_frame = state.frame.copy()
        
        if input_frame is None:
            time.sleep(0.1)
            continue

        # 2. เช็กคำสั่งจาก Web (Next / Timer)
        if state.command_queue:
            cmd = state.command_queue.popleft()
            if cmd == 'timer':
                ai.start_timer()
            elif cmd == 'next':
                if state.hn_queue:
                    state.hn_queue.rotate(-1)
                    state.current_hn = state.hn_queue[0]
                    print(f"⏩ Processing: {state.current_hn}")
                    ai.rx.reset()
                    ai.timer_result_text = ""
                    data = his.fetch_prescription(state.current_hn)
                    if data: ai.rx.update_from_his(data)

        # 3. ประมวลผล AI (กินเวลาเยอะสุดตรงนี้)
        # ส่งภาพเข้า AI Processor
        ai.latest_frame = input_frame 
        # (หมายเหตุ: ai.processor มี loop ของมันเอง หรือถ้าไม่มี ให้เรียก ai.process(input_frame) ตรงนี้เลยก็ได้)
        # แต่จากโค้ดเก่า ai มี thread แยก เราแค่ update frame ให้มัน
        
        # 4. วาด UI ลงบนภาพ (Drawing)
        # เราวาดทับลงบน input_frame เลย เพื่อเตรียมส่งขึ้นเว็บ
        display_frame = input_frame.copy()
        
        if ai.rx.is_ready:
            try:
                display_frame = ui.draw(display_frame, ai)
                
                # Logic Auto Reset
                if ai.rx.is_completed and (time.time() - ai.rx.complete_timestamp > 3.0):
                    ai.rx.reset()
                    ai.timer_result_text = ""
            except Exception as e:
                print(f"Draw Error: {e}")

        # 5. อัปเดตภาพผลลัพธ์กลับไปที่กองกลาง
        with state.lock:
            state.processed_frame = display_frame

        # AI ไม่ต้อง sleep เพราะมันช้าอยู่แล้ว รันเต็มสปีดเลย

# ==========================================
# 🌐 Thread 3: Web Server & Streaming (15 FPS)
# ==========================================
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>PillTrack Edge</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background: #111; color: #fff; font-family: sans-serif; text-align: center; padding: 20px; }
            .cam-container { position: relative; display: inline-block; border: 2px solid #444; }
            img { width: 100%; max-width: 800px; height: auto; display: block; }
            .btn-group { margin-top: 20px; display: flex; justify-content: center; gap: 15px; }
            button { padding: 15px 25px; font-size: 18px; border: none; border-radius: 8px; cursor: pointer; color: white; font-weight: bold;}
            .btn-next { background: #007bff; } .btn-timer { background: #28a745; }
            button:active { transform: scale(0.95); opacity: 0.8; }
        </style>
    </head>
    <body>
        <h2>💊 PillTrack Edge Monitor (Multithreaded)</h2>
        <div class="cam-container">
            <img src="/video_feed">
        </div>
        <div class="btn-group">
            <button class="btn-next" onclick="fetch('/cmd/next')">⏩ NEXT PATIENT</button>
            <button class="btn-timer" onclick="fetch('/cmd/timer')">⏱️ TIMER</button>
        </div>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    # Generator นี้จะทำงานแยกกันในแต่ละ Client ที่เปิดดู
    def generate():
        while state.running:
            # 1. หยิบภาพที่ผ่านการวาด UI แล้ว
            with state.lock:
                if state.processed_frame is None:
                    output_frame = None
                else:
                    output_frame = state.processed_frame.copy() # copy ออกมาจะได้ไม่ lock นาน

            if output_frame is None:
                time.sleep(0.1)
                continue

            # 2. Encode JPEG (กิน CPU พอสมควร)
            # แปลง RGB -> BGR ก่อนส่งขึ้นเว็บ
            out_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            (flag, encodedImage) = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            if not flag: continue

            # 3. ส่งข้อมูล
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            
            # 🔥 LIMIT FPS: หยุดรอให้ได้ประมาณ 15 FPS (1/15 = 0.066)
            # ช่วยลดภาระ CPU ในการ Encode JPEG ทำให้เอาแรงไปลงที่ AI ได้มากขึ้น
            time.sleep(0.06) 

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cmd/<action>')
def command(action):
    state.command_queue.append(action)
    return jsonify({"status": "ok"})

# ==========================================
# 🚀 Main Entry Point
# ==========================================
if __name__ == "__main__":
    # เริ่ม Thread กล้อง
    t_cam = threading.Thread(target=camera_worker, daemon=True)
    t_cam.start()

    # เริ่ม Thread AI
    t_ai = threading.Thread(target=ai_worker, daemon=True)
    t_ai.start()

    print(f"🌍 Server starting at http://0.0.0.0:5000")
    print(f"⚡ Mode: Multithreaded | Stream Limit: ~15 FPS")
    
    # รัน Web Server (Main Thread)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)