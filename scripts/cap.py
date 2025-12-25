import os
import cv2
import threading
import time
import numpy as np
from datetime import datetime
from flask import Flask, Response, jsonify, request

# ================= ⚙️ CONFIGURATION =================
DISPLAY_SIZE = (1280, 720) 
PORT = 5000
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_images")
# ====================================================

app = Flask(__name__)

# Global Variables
global_frame = None
current_save_path = None # ยังไม่มีโฟลเดอร์ จนกว่าจะกรอกหน้าเว็บ
lock = threading.Lock()
running = True

def camera_thread():
    """Thread เปิดกล้องค้างไว้"""
    global global_frame, running
    print("📷 Connecting to USB Camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_SIZE[1])

    if not cap.isOpened():
        print("❌ Error: เปิดกล้องไม่ติด!")
        running = False
        return

    print("✅ Camera Started!")
    
    while running:
        ret, frame = cap.read()
        if ret:
            with lock:
                global_frame = frame
        else:
            time.sleep(0.1)
    
    cap.release()

# ================= WEB ROUTES =================

@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Pill Collector</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background: #1e1e1e; color: #fff; font-family: sans-serif; text-align: center; padding: 10px; }
            
            /* Input Zone */
            .input-group { background: #2d2d2d; padding: 15px; border-radius: 10px; margin-bottom: 15px; }
            input { padding: 10px; font-size: 18px; border-radius: 5px; border: none; width: 60%; }
            .btn-set { padding: 10px 15px; font-size: 18px; background: #0984e3; color: white; border: none; border-radius: 5px; cursor: pointer; }
            
            /* Camera Zone */
            .cam-box { position: relative; border: 3px solid #444; display: inline-block; max-width: 100%; }
            img { width: 100%; max-width: 720px; height: auto; display: block; }
            
            /* Capture Button */
            .btn-capture {
                margin-top: 20px; width: 80%; padding: 20px; font-size: 24px; font-weight: bold;
                background: #636e72; color: #b2bec3; /* Disabled Color */
                border: none; border-radius: 50px; cursor: not-allowed; transition: 0.3s;
                box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            }
            .btn-capture.active {
                background: #ff4757; color: white; cursor: pointer;
            }
            .btn-capture.active:active { transform: scale(0.95); }

            /* Status Text */
            #current-label { color: #ffeaa7; font-weight: bold; margin-top: 5px; }
            #msg { margin-top: 10px; font-size: 14px; color: #aaa; }
        </style>
    </head>
    <body>
        <h2>💊 Pill Dataset Collector</h2>
        
        <div class="input-group">
            <input type="text" id="folderName" placeholder="Drug Name (e.g. para_500)">
            <button class="btn-set" onclick="setFolder()">SET</button>
            <div id="current-label">Waiting for input...</div>
        </div>

        <div class="cam-box">
            <img src="/video_feed" id="video">
        </div>
        
        <br>
        <button id="capBtn" class="btn-capture" onclick="capture()">🔒 LOCKED</button>
        <p id="msg"></p>

        <script>
            function setFolder() {
                let name = document.getElementById('folderName').value;
                if(!name) { alert("Please enter name!"); return; }
                
                fetch('/set_folder?name=' + name)
                    .then(res => res.json())
                    .then(data => {
                        document.getElementById('current-label').innerText = "📂 Current: " + data.path;
                        document.getElementById('current-label').style.color = "#55efc4";
                        
                        // ปลดล็อกปุ่มถ่าย
                        let btn = document.getElementById('capBtn');
                        btn.innerText = "📸 CAPTURE";
                        btn.classList.add('active');
                        btn.disabled = false;
                    });
            }

            function capture() {
                let btn = document.getElementById('capBtn');
                if(!btn.classList.contains('active')) return;

                fetch('/save_image')
                    .then(res => res.json())
                    .then(data => {
                        if(data.status === 'success') {
                            let msg = document.getElementById('msg');
                            msg.innerText = "✅ Saved: " + data.filename;
                            msg.style.color = "#2ed573";
                            
                            // Flash Effect
                            let vid = document.getElementById('video');
                            vid.style.opacity = 0.3;
                            setTimeout(() => vid.style.opacity = 1, 100);
                        } else {
                            alert("Error: " + data.message);
                        }
                    });
            }
        </script>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    def generate():
        while running:
            with lock:
                if global_frame is None: continue
                (flag, encodedImage) = cv2.imencode(".jpg", global_frame)
                if not flag: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            time.sleep(0.04) 
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_folder')
def set_folder():
    global current_save_path
    name = request.args.get('name', '').strip().replace(' ', '_')
    if not name: return jsonify({"status": "error", "message": "Invalid name"})
    
    # สร้าง Folder จริง
    current_save_path = os.path.join(BASE_DIR, name)
    os.makedirs(current_save_path, exist_ok=True)
    
    print(f"📂 Target set to: {current_save_path}")
    return jsonify({"status": "ok", "path": name})

@app.route('/save_image')
def save_image():
    global current_save_path
    if not current_save_path:
        return jsonify({"status": "error", "message": "No folder set!"})
    
    with lock:
        if global_frame is None: return jsonify({"status": "error"})
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img_{timestamp}.jpg"
        filepath = os.path.join(current_save_path, filename)
        
        cv2.imwrite(filepath, global_frame)
        print(f"💾 Saved: {filename} -> {os.path.basename(current_save_path)}")
        
        return jsonify({"status": "success", "filename": filename})

if __name__ == "__main__":
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()
    
    print(f"🚀 Web Collector Ready -> http://0.0.0.0:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)

