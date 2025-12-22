# export_dino_fix.py
import torch
import torch.nn as nn

# 1. สร้าง Wrapper Class เพื่อบังคับ Input ให้เหลือแค่ตัวเดียว
class DinoWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # เรียกโมเดลโดยไม่ส่ง masks ไป (ใช้ default ข้างในเอง)
        return self.model(x)

print("⏳ Downloading DINOv2...")
# โหลดโมเดลตัวเดิม
raw_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
raw_model.eval()

# 2. เอา Wrapper มาครอบ
model = DinoWrapper(raw_model)

# สร้าง Dummy Input
dummy_input = torch.randn(1, 3, 224, 224)

print("⏳ Exporting to ONNX (Clean Version)...")
torch.onnx.export(
    model, 
    dummy_input, 
    "dinov2_vitb14.onnx",  # ทับไฟล์เดิมเลย
    export_params=True, 
    opset_version=17, 
    do_constant_folding=True,
    input_names=['input'],   # ชื่อ Input ใน ONNX จะเหลือแค่อันนี้
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("✅ Done! Fixed 'masks' error.")