#!/usr/bin/env python3
import time
import cv2
import os
import json
from collections import deque
from dotenv import load_dotenv  
load_dotenv() 

# Import Modular Components
from src.hardware.camera import CameraHandler
from src.hardware.ui_renderer import UIRenderer
from src.ai.processor import AIProcessor
from src.services.his import HISConnector
from src.services.sync import SyncManager  # Import ตัว Sync

def main():
    # ==========================================
    # 1. ☁️ AUTO UPDATE (SMART SYNC)
    # ==========================================
    print("🔄 Connecting to S3 system...")
    try:
        syncer = SyncManager() 
        syncer.sync()
    except Exception as e:
        # ✅ 3. ปริ้นท์ Error สีแดงออกมาให้เห็นชัดๆ
        print(f"\n{'='*40}")
        print(f"❌ S3 SYNC FAILED: {e}")
        print(f"{'='*40}")
        print("💡 Hint: Check your .env file / Internet connection.")
        print("⚠️  Starting system with LOCAL files only...\n")

    # ==========================================
    # 2. Setup Components
    # ==========================================
    print("🚀 Initializing PillTrack Edge...")
    try:
        camera = CameraHandler()
        ai = AIProcessor().start()
        ui = UIRenderer()
        his = HISConnector()
    except Exception as e:
        print(f"❌ Init Error: {e}")
        return

    # 3. Setup Mock Data Queue
    MOCK_DB_PATH = "mock_server/prescriptions.json"
    hn_queue = deque(["HN123", "HN456"])
    if os.path.exists(MOCK_DB_PATH):
        try:
            with open(MOCK_DB_PATH, 'r') as f:
                hn_queue = deque(list(json.load(f).keys()))
        except: pass

    print("✅ System Ready. [Q] Quit | [N] Next Patient | [T] Timer")

    # 4. Main Loop
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        # AI Update
        ai.latest_frame = frame
        
        # UI Update
        display_frame = frame.copy()
        if ai.rx.is_ready:
            display_frame = ui.draw(display_frame, ai)
            
            # Auto Reset Logic
            if ai.rx.is_completed and (time.time() - ai.rx.complete_timestamp > 3.0):
                ai.rx.reset()
                ai.timer_result_text = ""
        # else:
        #     status = f"NEXT: {hn_queue[0]}" if hn_queue else "NO DATA"
        #     cv2.putText(display_frame, f"PRESS 'N' FOR {status}", (380, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        #
        cv2.imshow("PillTrack Edge", display_frame)

        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('t'):
            ai.start_timer()
        elif key == ord('n'):
            if hn_queue:
                hn_queue.rotate(-1)
                curr_hn = hn_queue[0]
                print(f"⏩ Next: {curr_hn}")
                ai.rx.reset()
                ai.timer_result_text = ""
                data = his.fetch_prescription(curr_hn)
                if data: ai.rx.update_from_his(data)

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()