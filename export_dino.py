import torch
import torch.nn as nn

# โหลด DINOv2
print("⏳ Downloading DINOv2...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()

# สร้าง Dummy Input (ขนาดเท่าที่ใช้จริง)
dummy_input = torch.randn(1, 3, 224, 224)

# Export เป็น ONNX
print("⏳ Exporting to ONNX (This may take a minute)...")
torch.onnx.export(
    model, 
    dummy_input, 
    "dinov2_vits14.onnx", 
    export_params=True, 
    opset_version=17, 
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("✅ Done! Saved 'dinov2_vits14.onnx'")