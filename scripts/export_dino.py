import torch
import torch.nn as nn
import os
import yaml
import sys

# ==========================================
# 1. Wrapper Class
# ==========================================
class DinoWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # DINOv2 จาก Torch Hub คืนค่าเป็น Tensor (Batch, Embed_Dim) อยู่แล้ว
        return self.model(x)

def export_model():
    print("🚀 Starting DINOv2 Export Process...")

    # ==========================================
    # 2. Setup Paths & Load Config
    # ==========================================
    # หาตำแหน่ง Root Project (ถอยจาก scripts/ หรือที่วางไฟล์นี้ 1 ชั้น)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) 
    # ถ้าไฟล์นี้วางที่ root อยู่แล้ว ให้ใช้: project_root = script_dir
    
    config_path = os.path.join(project_root, "config.yaml")
    
    # ค่า Default
    ai_size = 336 

    # อ่านไฟล์ config.yaml
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                # ดึงค่าตามโครงสร้าง: settings -> ai_size
                ai_size = cfg.get('settings', {}).get('ai_size', 336)
            print(f"📖 Loaded Config from {config_path}")
            print(f"🎯 Target AI Size: {ai_size}x{ai_size}")
        except Exception as e:
            print(f"⚠️ Error loading config: {e}. Using default {ai_size}.")
    else:
        print(f"⚠️ Config file not found at {config_path}. Using default {ai_size}.")

    # ==========================================
    # 3. Load & Prepare Model
    # ==========================================
    print("⏳ Downloading DINOv2 (ViT-B/14) from Torch Hub...")
    # หมายเหตุ: ถ้าเน็ตช้า หรืออยากได้ตัวเล็ก ให้เปลี่ยนเป็น 'dinov2_vits14' (Small)
    raw_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    raw_model.eval()
    
    model = DinoWrapper(raw_model)

    # 🔥 SENIOR TRICK: ใช้ Batch Size = 2 เพื่อย้ำว่าเป็น Dynamic Batch
    dummy_input = torch.randn(2, 3, ai_size, ai_size)
    
    # เตรียม Output Path
    output_dir = os.path.join(project_root, "models")
    output_file = os.path.join(output_dir, "dinov2_vitb14.onnx")
    os.makedirs(output_dir, exist_ok=True)

    print(f"📦 Exporting to ONNX at: {output_file}...")
    
    # ==========================================
    # 4. Export
    # ==========================================
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            output_file,
            export_params=True, 
            opset_version=14,        # แนะนำ 17 สำหรับ Transformer ยุคใหม่
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # ✅ พระเอกของเรา: บอกว่าแกน 0 (Batch) ยืดหดได้
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print("-" * 50)
        print(f"✅ SUCCESS! Model Exported.")
        print(f"📂 Location: {output_file}")
        print(f"📐 Input Size: {ai_size}x{ai_size}")
        print(f"🔄 Dynamic Batch: Enabled (Ready for batch processing)")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Export Failed: {e}")

if __name__ == "__main__":
    export_model()