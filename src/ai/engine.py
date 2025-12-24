import onnxruntime as ort
import numpy as np
import cv2
import os
from typing import List
from src.utils.config import CFG  

class FeatureEngine:
    def __init__(self):
        print(f"⏳ Loading DINOv2 (Expected Input: {CFG.AI_SIZE}x{CFG.AI_SIZE})...")
        try:
            model_path = "models/dinov2_vitb14.onnx"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            # Load ONNX model
            self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.sess.get_inputs()[0].name
            
            # Print debug info
            print(f"✅ DINOv2 Loaded! Input Name: {self.input_name}")
            
        except Exception as e:
            print(f"❌ Error loading DINO ONNX: {e}")
            self.sess = None
            
    def preprocess_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        # สร้าง Array รอตามจำนวนภาพ
        batch = np.zeros((len(crop_list), 3, CFG.AI_SIZE, CFG.AI_SIZE), dtype=np.float32)
        
        for i, img in enumerate(crop_list):
            # 1. Resize เป็น 336x336
            img_resized = cv2.resize(img, (CFG.AI_SIZE, CFG.AI_SIZE), interpolation=cv2.INTER_LINEAR)
            
            # 2. Normalize
            img_norm = (img_resized.astype(np.float32) / 255.0 - CFG.MEAN) / CFG.STD
            
            # 3. HWC -> CHW (3, 336, 336)
            batch[i] = img_norm.transpose(2, 0, 1)
            
        return batch

    def extract_dino_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        if not crop_list or self.sess is None: return np.array([])
        
        # 🛡️ SAFE MODE: Process ทีละรูป เพื่อกัน ONNX Batch Error
        embeddings_list = []
        
        try:
            for crop in crop_list:
                # 1. เตรียมภาพเดี่ยว (Batch Size = 1)
                # เราส่ง list ที่มีรูปเดียวไปเข้า preprocess
                single_batch = self.preprocess_batch([crop])
                
                # 2. ส่งเข้า ONNX ทีละใบ
                outputs = self.sess.run(None, {self.input_name: single_batch})
                
                # 3. เก็บผลลัพธ์
                # outputs[0] จะได้ shape (1, 768)
                embeddings_list.append(outputs[0])

            # ถ้าไม่มีผลลัพธ์เลย
            if not embeddings_list: return np.array([])

            # 4. รวมร่างกลับมาเป็นก้อนเดียว (N, 768)
            embeddings = np.vstack(embeddings_list)
            
            # 5. L2 Normalization (ทำทีเดียวตอนจบ)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-6)
            
            return embeddings

        except Exception as e:
            print(f"❌ Inference Error: {e}")
            # Return empty array to prevent crash
            return np.array([])