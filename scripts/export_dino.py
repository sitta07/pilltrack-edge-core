import torch
import torch.nn as nn
import os
import yaml  # ✅ เพิ่ม import yaml

# ==========================================
# 1. Wrapper Class
# ==========================================
class DinoWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

def export_model():
    # ==========================================
    # 2. Setup Paths & Load Config
    # ==========================================
    # หาตำแหน่ง Root Project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, "config.yaml")
    
    # ค่า Default เผื่อหาไฟล์ไม่เจอ
    ai_size = 336

    # อ่านไฟล์ config.yaml
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                # ดึงค่า ai_size จาก settings
                ai_size = cfg.get('settings', {}).get('ai_size', 336)
            print(f"📖 Loaded Config: Using AI Size = {ai_size}x{ai_size}")
        except Exception as e:
            print(f"⚠️ Error loading config: {e}. Using default {ai_size}.")
    else:
        print(f"⚠️ Config file not found at {config_path}. Using default {ai_size}.")

    # ==========================================
    # 3. Load & Prepare Model
    # ==========================================
    print("⏳ Downloading DINOv2 (ViT-B/14)...")
    raw_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    raw_model.eval()
    model = DinoWrapper(raw_model)

    # ✅ ใช้ ai_size ที่อ่านมา สร้าง Dummy Input
    dummy_input = torch.randn(1, 3, ai_size, ai_size)
    
    # เตรียม Output Path
    output_dir = os.path.join(project_root, "models")
    output_file = os.path.join(output_dir, "dinov2_vitb14.onnx")
    os.makedirs(output_dir, exist_ok=True)

    print(f"⏳ Exporting to {output_file}...")
    
    # ==========================================
    # 4. Export
    # ==========================================
    torch.onnx.export(
        model, 
        dummy_input, 
        output_file,
        export_params=True, 
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✅ Success! Exported model size: {ai_size}x{ai_size}")
    print(f"📂 Saved at: {output_file}")

if __name__ == "__main__":
    export_model()