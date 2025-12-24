import os
import cv2
import threading
import time
import numpy as np
from datetime import datetime
from flask import Flask, Response, jsonify

# ================= ⚙️ CONFIGURATION =================
# แก้ไขความละเอียดกล้องตรงนี้
DISPLAY_SIZE = (1280, 720) 
PORT = 5000
# ====================================================

app = Flask(__name__)

# Global Variables
global_frame = None
save_folder = ""
lock = threading.Lock()
running = True

def ask_folder_name():
    """ถามชื่อโฟลเดอร์ผ่าน Terminal"""
    while True:
        print("\n" + "="*40)
        name = input("📂 กรุณาใส่ชื่อ Folder เก็บภาพ (เช่น pills_paracap): ").strip()
        if name:
            name = name.replace(" ", "_")
            # สร้าง path จริง
            current_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(current_dir, "captured_images", name)
            os.makedirs(full_path, exist_ok=True)
            return full_path, name
        print("⚠️ ชื่อห้ามว่าง! เอาใหม่")

def camera_thread():
    """Thread สำหรับอ่านกล้องตลอดเวลา"""
    global global_frame, running
    
    print("📷 Connecting to USB Camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_SIZE[1])

    if not cap.isOpened():
        print("❌ Error: เปิดกล้องไม่ติด! เช็คสาย USB หน่อย")
        running = False
        return

    print("✅ Camera Started! Ready to stream.")
    
    while running:
        ret, frame = cap.read()
        if ret:
            with lock:
                global_frame = frame
        else:
            time.sleep(0.1)
    
    cap.release()
    print("📷 Camera Stopped")

# ================= WEB ROUTES =================

@app.route('/')
def index():
    return f"""
    <html>
    <head>
        <title>Data Collector</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ background: #222; color: white; font-family: sans-serif; text-align: center; padding: 20px; }}
            h2 {{ color: #00d2ff; }}
            .container {{ display: inline-block; position: relative; border: 3px solid #444; }}
            img {{ width: 100%; max-width: 800px; height: auto; display: block; }}
            .overlay {{ 
                position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                width: 50px; height: 50px; border: 2px solid rgba(0,255,0,0.5); border-radius: 50%;
                pointer-events: none;
            }}
            .btn-capture {{
                margin-top: 20px; padding: 20px 50px; font-size: 24px;
                background: #ff4757; color: white; border: none; border-radius: 50px;
                cursor: pointer; box-shadow: 0 5px 15px rgba(255, 71, 87, 0.4);
            }}
            .btn-capture:active {{ transform: scale(0.95); background: #ff6b81; }}
            .status {{ margin-top: 10px; color: #ccc; }}
        </style>
    </head>
    <body>
        <h2>📂 Saving to: {os.path.basename(save_folder)}</h2>
        <div class="container">
            <img src="/video_feed" id="video">
            <div class="overlay">+</div>
        </div>
        <br>
        <button class="btn-capture" onclick="capture()">📸 CAPTURE</button>
        <p class="status" id="msg">Ready...</p>

        <script>
            function capture() {{
                fetch('/save_image')
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('msg').innerText = "✅ " + data.filename;
                        document.getElementById('msg').style.color = "#2ed573";
                        
                        // Flash effect
                        let img = document.getElementById('video');
                        img.style.opacity = 0.5;
                        setTimeout(() => img.style.opacity = 1, 100);
                    }});
            }}
        </script>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    def generate():
        while running:
            with lock:
                if global_frame is None:
                    continue
                # Encode เป็น JPEG ส่งขึ้นเว็บ
                (flag, encodedImage) = cv2.imencode(".jpg", global_frame)
                if not flag: continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
            time.sleep(0.03) # Limit FPS นิดหน่อย
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_image')
def save_image():
    with lock:
        if global_frame is None:
            return jsonify({"status": "error"})
        
        # Save Frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img_{timestamp}.jpg"
        filepath = os.path.join(save_folder, filename)
        
        cv2.imwrite(filepath, global_frame)
        print(f"💾 Saved: {filename}")
        
        return jsonify({"status": "success", "filename": filename})

if __name__ == "__main__":
    try:
        # 1. ถามชื่อโฟลเดอร์ก่อน
        save_folder, folder_name = ask_folder_name()
        print(f"📁 Image Path: {save_folder}")
        
        # 2. เริ่มกล้องใน Thread แยก
        t = threading.Thread(target=camera_thread, daemon=True)
        t.start()

        # 3. เริ่ม Web Server
        print(f"\n🚀 เปิดเว็บเพื่อถ่ายรูปได้ที่ -> http://0.0.0.0:{PORT}")
        print("กด Ctrl+C เพื่อจบโปรแกรม")
        
        app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)

    except KeyboardInterrupt:
        running = False
        print("\n👋 Exiting...")