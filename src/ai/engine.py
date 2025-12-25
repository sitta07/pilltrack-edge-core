import onnxruntime as ort
import numpy as np
import cv2
import os
from typing import List
from src.utils.config import CFG  

class FeatureEngine:
    def __init__(self):
        # ใช้ CFG.AI_SIZE ตามที่คุณต้องการ (เช่น 336)
        print(f"⏳ Loading DINOv2 (Target Input: {CFG.AI_SIZE}x{CFG.AI_SIZE})...")
        try:
            model_path = "models/dinov2_vitb14.onnx"
            if not os.path.exists(model_path):
                print(f"⚠️ Warning: Model not found at {model_path}")

            # Load ONNX model
            self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.sess.get_inputs()[0].name
            
            # Print debug info
            print(f"✅ DINOv2 Loaded! Input Name: {self.input_name}")
            
            # Pre-calculate Constants for Speed (ImageNet Mean/Std)
            # Shape (1, 1, 3) เพื่อให้ Broadcast กับภาพ (H, W, 3) ได้เลย
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
            
        except Exception as e:
            print(f"❌ Error loading DINO ONNX: {e}")
            self.sess = None
            
    def preprocess_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        # 1. จองพื้นที่หน่วยความจำรอไว้เลย (Batch Size, 3, H, W)
        # ใช้ CFG.AI_SIZE ตรงนี้ตาม request
        batch = np.zeros((len(crop_list), 3, CFG.AI_SIZE, CFG.AI_SIZE), dtype=np.float32)
        
        for i, img in enumerate(crop_list):
            # Resize
            img_resized = cv2.resize(img, (CFG.AI_SIZE, CFG.AI_SIZE), interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR -> RGB (DINO ต้องการ RGB)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize (0-1)
            img_norm = img_rgb.astype(np.float32) / 255.0
            
            # Standardize (ImageNet stats)
            img_norm = (img_norm - self.mean) / self.std
            
            # HWC -> CHW (Transpose)
            batch[i] = img_norm.transpose(2, 0, 1)
            
        return batch

    def extract_dino_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        if not crop_list or self.sess is None: return np.array([])
        
        try:
            # ⚡ 1. Preprocess ทีเดียวทั้ง Batch
            batch_input = self.preprocess_batch(crop_list)
            
            # ⚡ 2. ONNX Runtime Inference (จุดสำคัญที่ทำให้เร็ว!)
            # ส่งไปคำนวณรอบเดียวจบ ไม่วนลูป run() แล้ว
            outputs = self.sess.run(None, {self.input_name: batch_input})
            
            # DINO Output shape: (Batch_Size, Embed_Dim)
            embeddings = outputs[0]

            # 3. L2 Normalization (Vectorized)
            # สำคัญมากสำหรับการทำ Search ด้วย Dot Product
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-6)
            
            return embeddings

        except Exception as e:
            print(f"❌ Inference Error: {e}")
            return np.array([])