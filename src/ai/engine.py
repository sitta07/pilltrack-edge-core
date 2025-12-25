import onnxruntime as ort
import numpy as np
import cv2
import os
from typing import List
from src.utils.config import CFG  

class FeatureEngine:
    def __init__(self):
        self.DINO_SIZE = 224 
        
        print(f"⏳ Loading DINOv2 (Target Input: {self.DINO_SIZE}x{self.DINO_SIZE})...")
        try:
            # 💡 แนะนำ: ลองหาไฟล์ 'dinov2_vits14.onnx' (Small) มาใช้แทน vitb14 (Base)
            model_path = "models/dinov2_vitb14.onnx" 
            if not os.path.exists(model_path):
                # Fallback หรือแจ้งเตือน
                print(f"⚠️ Warning: Model not found at {model_path}")

            # Load ONNX model
            self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.sess.get_inputs()[0].name
            
            # Print debug info
            print(f"✅ DINOv2 Loaded! Input Name: {self.input_name}")
            
            # Pre-calc constants for speed
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
            
        except Exception as e:
            print(f"❌ Error loading DINO ONNX: {e}")
            self.sess = None
            
    def preprocess_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        # เตรียม Array ก้อนใหญ่ (N, 3, 224, 224)
        batch = np.zeros((len(crop_list), 3, self.DINO_SIZE, self.DINO_SIZE), dtype=np.float32)
        
        for i, img in enumerate(crop_list):
            # 1. Resize (ใช้ 224 Fix ไปเลย เพื่อความเร็ว)
            img_resized = cv2.resize(img, (self.DINO_SIZE, self.DINO_SIZE), interpolation=cv2.INTER_LINEAR)
            
            # 2. Normalize & Standardize (Vectorized Operation เร็วกว่าหารทีละตัว)
            # แปลง BGR -> RGB (สำคัญมาก! DINO เทรนด้วย RGB)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize 0-1
            img_norm = img_rgb.astype(np.float32) / 255.0
            
            # Standardize (img - mean) / std
            img_norm = (img_norm - self.mean) / self.std
            
            # 3. HWC -> CHW
            batch[i] = img_norm.transpose(2, 0, 1)
            
        return batch

    def extract_dino_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        if not crop_list or self.sess is None: return np.array([])
        
        try:
            # ⚡ 1. Preprocess ทีเดียวทั้ง Batch
            batch_input = self.preprocess_batch(crop_list)
            
            # ⚡ 2. Inference ทีเดียวทั้งก้อน (One Shot Inference)
            # นี่คือจุดที่ลดเวลาจาก 3000ms เหลือ 300ms
            outputs = self.sess.run(None, {self.input_name: batch_input})
            
            # outputs[0] Shape: (Batch_Size, 768) หรือ (Batch_Size, 384) แล้วแต่รุ่น
            embeddings = outputs[0]

            # 3. L2 Normalization (Vectorized)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-6)
            
            return embeddings

        except Exception as e:
            print(f"❌ Inference Error: {e}")
            return np.array([])