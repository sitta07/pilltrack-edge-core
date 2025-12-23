import onnxruntime as ort
import numpy as np
import cv2
from typing import List
from src.utils.config import CFG

class FeatureEngine:
    def __init__(self):
        print("⏳ Loading DINOv2 (ONNX Runtime)...")
        try:
            # Load ONNX model
            self.sess = ort.InferenceSession("models/dinov2_vitb14.onnx", providers=['CPUExecutionProvider'])
            self.input_name = self.sess.get_inputs()[0].name
        except Exception as e:
            print(f"❌ Error loading DINO ONNX: {e}")
            self.sess = None
            
    def preprocess_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        batch = np.zeros((len(crop_list), 3, 224, 224), dtype=np.float32)
        for i, img in enumerate(crop_list):
            img_rgb = img[:, :, :3] 
            img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            # Normalize
            img_norm = (img_resized.astype(np.float32) / 255.0 - CFG.MEAN) / CFG.STD
            batch[i] = img_norm.transpose(2, 0, 1) # HWC -> CHW
        return batch

    def extract_dino_batch(self, crop_list: List[np.ndarray]) -> np.ndarray:
        if not crop_list or self.sess is None: return np.array([])
        
        img_batch_np = self.preprocess_batch(crop_list)
        outputs = self.sess.run(None, {self.input_name: img_batch_np})
        
        embeddings = outputs[0]
        # L2 Normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-6)
        return embeddings