from ultralytics import YOLO

# --- Config ---
MODEL_PATH = "models/seg_best_process.pt"  # เปลี่ยนเป็น path ไฟล์ .pt ของคุณ
EXPORT_SIZE = 640       # ควรตรงกับ CFG.AI_SIZE ในโค้ดรันจริง (เช่น 320, 416, 640)

def main():
    print(f"🚀 Loading model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    print("📦 Starting Export to ONNX...")
    
    path = model.export(
        format="onnx",
        imgsz=EXPORT_SIZE,
        opset=12,
        simplify=True
    )

    print(f"✅ Export Completed! Saved at: {path}")
    print(f"💡 Tip: อย่าลืมไปแก้ CFG.MODEL_PACK ในโค้ดหลักให้ชี้มาที่ไฟล์นี้นะครับ")

if __name__ == "__main__":
    main()