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
            # Check path to make sure it exists
            model_path = "models/dinov2_vitb14.onnx"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            # Load ONNX model
            self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.sess.get_inputs()[0].name
        except Exception as e:
            print(f"❌ Error loading DINO ONNX: {e}")
            self.sess = None
            
    def preprocess_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        # 🛠️ FIX 1: ใช้ CFG.AI_SIZE แทน 224 ในการสร้าง array เปล่า
        batch = np.zeros((len(crop_list), 3, CFG.AI_SIZE, CFG.AI_SIZE), dtype=np.float32)
        
        for i, img in enumerate(crop_list):
            img_rgb = img[:, :, :3] 
            
            # 🛠️ FIX 2: Resize รูปให้เป็นขนาดตาม Config (เช่น 336x336)
            img_resized = cv2.resize(img_rgb, (CFG.AI_SIZE, CFG.AI_SIZE), interpolation=cv2.INTER_LINEAR)
            
            # Normalize
            img_norm = (img_resized.astype(np.float32) / 255.0 - CFG.MEAN) / CFG.STD
            batch[i] = img_norm.transpose(2, 0, 1) # HWC -> CHW
            
        return batch

    def extract_dino_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        if not crop_list or self.sess is None: return np.array([])
        
        img_batch_np = self.preprocess_batch(crop_list)
        
        try:
            outputs = self.sess.run(None, {self.input_name: img_batch_np})
            embeddings = outputs[0]
            # L2 Normalization
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-6)
            return embeddings
        except Exception as e:
            print(f"❌ Inference Error: {e}")
            return np.array([])