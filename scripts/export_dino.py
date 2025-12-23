import torch
import torch.nn as nn
import os

# ==========================================
# 1. Wrapper Class (หัวใจสำคัญในการแก้บั๊ก Masks)
# ==========================================
class DinoWrapper(nn.Module):
    """
    Wrapper นี้ทำหน้าที่ 'ซ่อน' argument ที่ไม่จำเป็น (เช่น masks)
    เพื่อให้ ONNX เห็น Input แค่ตัวเดียวคือ 'x' (รูปภาพ)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # บังคับเรียก model ด้วย input ตัวเดียว
        # DINOv2 จะไปจัดการ default value ของ masks เองข้างใน
        return self.model(x)

def export_model():
    print("⏳ Downloading DINOv2 (ViT-B/14)...")
    
    # โหลดโมเดลจาก PyTorch Hub
    # เลือกขนาดได้: dinov2_vits14 (Small), dinov2_vitb14 (Base), dinov2_vitl14 (Large)
    raw_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    raw_model.eval()

    # เอา Wrapper มาครอบ
    model = DinoWrapper(raw_model)

    dummy_input = torch.randn(1, 3, 336, 336)
    
    # ==========================================
    # 2. Path Handling (แก้เรื่อง No such file)
    # ==========================================
    # หาตำแหน่งของไฟล์ script นี้ (scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ถอยหลัง 1 ก้าวเพื่อหา Project Root (RASP_PROJECT/)
    project_root = os.path.dirname(script_dir)
    
    # สร้าง Path ไปยังโฟลเดอร์ models ที่ Root
    output_dir = os.path.join(project_root, "models")
    output_file = os.path.join(output_dir, "dinov2_vitb14.onnx")

    # สร้างโฟลเดอร์ models เผื่อไว้ (ถ้ายังไม่มี)
    os.makedirs(output_dir, exist_ok=True)

    print(f"⏳ Exporting to {output_file} (Clean Version)...")
    
    # ==========================================
    # 3. Export to ONNX
    # ==========================================
    torch.onnx.export(
        model, 
        dummy_input, 
        output_file,
        export_params=True, 
        opset_version=17,       # แนะนำ 17 ขึ้นไปสำหรับ Transformer
        do_constant_folding=True,
        input_names=['input'],  # ชื่อตัวแปร Input ในไฟล์ ONNX
        output_names=['output'], # ชื่อตัวแปร Output
        dynamic_axes={
            'input': {0: 'batch_size'},  # รองรับ Batch size ยืดหยุ่นได้
            'output': {0: 'batch_size'}
        }
    )

    print(f"✅ Success! File saved at: {output_file}")
    print("👉 Next Step: Run 'main.py' to use this model.")

if __name__ == "__main__":
    export_model()