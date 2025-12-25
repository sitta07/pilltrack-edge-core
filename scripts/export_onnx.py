import os
from ultralytics import YOLO

# 1. หาตำแหน่ง Root ของโปรเจกต์แบบ Auto
# หา path ของไฟล์นี้ (scripts/convert.py)
current_script_path = os.path.abspath(__file__)
# ถอยออกมา 1 ชั้นเพื่อเจอ folder 'scripts'
script_dir = os.path.dirname(current_script_path)
# ถอยอีก 1 ชั้นเพื่อเจอ 'Project_Root'
project_root = os.path.dirname(script_dir)

# 2. ประกอบร่าง Path ไปหาไฟล์โมเดล
# มันจะกลายเป็น: .../Project_Root/models/seg_best_process.pt
model_path = os.path.join(project_root, 'models', 'seg_best_process.pt')

print(f"📍 Script location: {script_dir}")
print(f"🎯 Target Model Path: {model_path}")

# เช็คความชัวร์ก่อนรัน
if not os.path.exists(model_path):
    print(f"\n❌ Error: หาไฟล์โมเดลไม่เจอที่: {model_path}")
    print("👉 ลองเช็คชื่อไฟล์ หรือโครงสร้าง Folder อีกทีนะครับ")
    exit()

# 3. โหลดและ Export
print(f"\n🚀 Loading model...")
model = YOLO(model_path)

print("📦 Exporting to ONNX...")
output_path = model.export(
    format="onnx",
    imgsz=640,       # ตั้งให้ตรงกับที่ใช้จริง (เช่น 640)
    opset=12,
    simplify=True
)

print("-" * 50)
print(f"✅ Export เสร็จเรียบร้อย!")
print(f"📂 ไฟล์ ONNX ถูกเซฟไว้ที่: {output_path}")
print("-" * 50)