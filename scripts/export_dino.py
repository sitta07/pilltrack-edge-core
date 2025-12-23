import torch
import torch.nn as nn
import os

class DinoWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

def export_model():
    print("⏳ Downloading DINOv2 (ViT-B/14)...")
    
  
    raw_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    raw_model.eval()

    # เอา Wrapper มาครอบ
    model = DinoWrapper(raw_model)

    # สร้าง Dummy Input (ขนาดต้องตรงกับที่ใช้จริงคือ 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    output_file = "models/dinov2_vitb14.onnx"

    print(f"⏳ Exporting to {output_file} (Clean Version)...")
    
    # Export เป็น ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        output_file,
        export_params=True, 
        opset_version=17,       # แนะนำ 17 หรือสูงกว่าสำหรับ Transformer
        do_constant_folding=True,
        input_names=['input'],  # ชื่อตัวแปร Input ในไฟล์ ONNX
        output_names=['output'], # ชื่อตัวแปร Output
        dynamic_axes={
            'input': {0: 'batch_size'},  # รองรับ Batch size ยืดหยุ่นได้
            'output': {0: 'batch_size'}
        }
    )

    print(f"✅ Success! File saved at: {os.path.abspath(output_file)}")
    print("👉 Next Step: Upload this file to S3 or copy to 'models/' folder on Raspberry Pi.")

if __name__ == "__main__":
    export_model()