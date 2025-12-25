import os
from ultralytics import YOLO

# 1. ระบุ Path ของไฟล์ .pt
model_path = "models/seg_best_process.pt"

# เช็คก่อนว่าไฟล์มีอยู่จริงไหม
if not os.path.exists(model_path):
    print(f"❌ ไม่เจอไฟล์ที่: {model_path}")
    print("👉 ตรวจสอบว่าโฟลเดอร์ 'models' และไฟล์ถูกต้องนะครับ")
    exit()

print(f"🚀 Loading model from: {model_path}")
model = YOLO(model_path)

print("📦 Exporting to ONNX...")

# 2. สั่ง Export
# imgsz=640 : ควรตั้งให้ตรงกับ CFG.AI_SIZE ในโค้ดรันจริงของคุณ
# simplify=True : ช่วยลดขนาดไฟล์และทำให้รันเร็วขึ้น
path = model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True
)

print("-" * 50)
print(f"✅ Export เสร็จสมบูรณ์!")
print(f"📂 ไฟล์ ONNX อยู่ที่: {path}")
print("-" * 50)