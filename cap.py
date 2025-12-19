import time
import os
import cv2
import numpy as np
from datetime import datetime

try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ Error: ไม่พบ library 'picamera2'")
    exit()

# ================= ⚙️ CONFIGURATION =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SAVE_FOLDER = os.path.join(CURRENT_DIR, "captured_images")
DISPLAY_SIZE = (1280, 720)
# ====================================================


def ask_folder_name():
    """
    ให้ user ใส่ชื่อ folder ก่อนเริ่ม capture
    """
    while True:
        folder_name = input("📂 กรุณาใส่ชื่อ folder สำหรับเก็บภาพ: ").strip()

        if folder_name == "":
            print("⚠️ ชื่อ folder ห้ามว่าง ลองใหม่อีกครั้ง")
            continue

        # ป้องกัน path แปลก ๆ
        folder_name = folder_name.replace(" ", "_")
        return folder_name


def main():
    # ====== ASK USER FIRST ======
    user_folder = ask_folder_name()
    SAVE_FOLDER = os.path.join(BASE_SAVE_FOLDER, user_folder)

    os.makedirs(SAVE_FOLDER, exist_ok=True)

    print("\n📁 Save Folder:", SAVE_FOLDER)
    print("-" * 40)

    print("📷 Initializing Picamera2...")
    picam = Picamera2()

    config = picam.create_preview_configuration(
        main={"size": DISPLAY_SIZE, "format": "RGB888"}
    )
    picam.configure(config)
    picam.start()

    print("✅ System Ready!")
    print("   [Mode]: รับภาพ RGB")
    print("   [Key] : P = Capture | Q = Quit")
    print("-" * 40)

    try:
        while True:
            frame_rgb = picam.capture_array()
            frame_preview = frame_rgb.copy()

            # Draw crosshair
            h, w = frame_preview.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.line(frame_preview, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
            cv2.line(frame_preview, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)

            cv2.imshow("Data Collector", frame_preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('p') or key == ord('P'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"img_{timestamp}.jpg"
                filepath = os.path.join(SAVE_FOLDER, filename)

                try:
                    cv2.imwrite(filepath, frame_rgb)
                    print(f"📸 Saved: {filename}")

                    # Flash effect
                    flash = np.ones(frame_preview.shape, dtype=np.uint8) * 255
                    cv2.imshow(
                        "Data Collector",
                        cv2.addWeighted(frame_preview, 0.5, flash, 0.5, 0),
                    )
                    cv2.waitKey(50)

                except Exception as e:
                    print(f"❌ Save Error: {e}")

            elif key == ord('q') or key == ord('Q'):
                break

    except Exception as e:
        print(f"⚠️ Error: {e}")

    finally:
        picam.stop()
        cv2.destroyAllWindows()
        print("👋 Exit Program")


if __name__ == "__main__":
    main()
