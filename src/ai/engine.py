import onnxruntime as ort
import numpy as np
import cv2
import os
from typing import List
from src.utils.config import CFG  

class FeatureEngine:
    def __init__(self):
        # ใช้ขนาดจาก Config ให้ตรงกับตอน Export
        self.input_size = CFG.AI_SIZE
        
        print(f"⏳ Loading DINOv2 ONNX (Input: {self.input_size}x{self.input_size})...")
        
        try:
            # path ไฟล์โมเดล (ต้องตรงกับที่ Export มา)
            model_path = "models/dinov2_vitb14.onnx"
            
            if not os.path.exists(model_path):
                print(f"⚠️ Warning: Model not found at {model_path}")

            # 1. Load ONNX Runtime
            # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] # ถ้ามี GPU ให้ใส่ CUDA นำหน้า
            self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            
            # ดึงชื่อ Input node จากโมเดล (กันพลาด)
            self.input_name = self.sess.get_inputs()[0].name
            
            print(f"✅ DINOv2 Engine Ready! (Batch Support Enabled)")
            
            # 2. Pre-calculate Constants (ImageNet Mean/Std)
            # เตรียม shape (1, 1, 3) เพื่อให้ Broadcast กับภาพได้เร็วๆ
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
            
        except Exception as e:
            print(f"❌ Error loading DINO Engine: {e}")
            self.sess = None
            
    def preprocess_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        """
        แปลง List ของภาพ (BGR) เป็น Numpy Batch (N, 3, H, W) ที่ Normalize แล้ว
        """
        num_imgs = len(crop_list)
        
        # จอง Memory ทีเดียว (Batch Size, 3, H, W)
        batch = np.zeros((num_imgs, 3, self.input_size, self.input_size), dtype=np.float32)
        
        for i, img in enumerate(crop_list):
            # 1. Resize ให้ตรงกับ model input
            img_resized = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
            
            # 2. Convert BGR -> RGB (DINO ต้องใช้ RGB)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # 3. Normalize (0-1)
            img_norm = img_rgb.astype(np.float32) / 255.0
            
            # 4. Standardize (ImageNet stats) -> (Pixel - Mean) / Std
            img_norm = (img_norm - self.mean) / self.std
            
            # 5. Transpose HWC -> CHW (3, H, W)
            batch[i] = img_norm.transpose(2, 0, 1)
            
        return batch

    def extract_dino_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        """
        รัน Inference ทีเดียวทั้ง Batch (เร็วที่สุด)
        """
        if not crop_list or self.sess is None: 
            return np.array([])
        
        try:
            # ⚡ 1. Preprocess รวดเดียว
            batch_input = self.preprocess_batch(crop_list)
            
            # ⚡ 2. ONNX Runtime Inference (One Shot)
            # ส่งไปคำนวณรอบเดียวจบ ไม่ต้องวน Loop แล้ว เพราะโมเดลรองรับ Dynamic Batch
            outputs = self.sess.run(None, {self.input_name: batch_input})
            
            # Output Shape: (Batch_Size, Embed_Dim) เช่น (10, 768)
            embeddings = outputs[0]

            # 3. L2 Normalization (Vectorized)
            # ทำให้ Vector เป็น Unit Vector เพื่อให้ Dot Product = Cosine Similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-6)
            
            return embeddings

        except Exception as e:
            print(f"❌ DINO Inference Error: {e}")
            # ถ้า Batch ใหญ่ไปจนเมมเต็ม อาจต้อง fallback ไปทำทีละรูป (แต่ปกติ Text/Image ไม่ค่อยเต็มง่ายๆ)
            return np.array([])