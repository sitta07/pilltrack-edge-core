import yaml
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

# Load YAML
try:
    with open("config.yaml", "r") as f:
        yaml_cfg = yaml.safe_load(f)
except FileNotFoundError:
    # Default config fallback
    yaml_cfg = {
        'artifacts': {
            'model': 'models/seg_best_process.pt', 
            'drug_list': 'database/drug_list.json',
            'pack_vec': 'database/pill_fingerprints.pkl'
        },
        'display': {'width': 1280, 'height': 720},
        'settings': {'yolo_conf': 0.5},
        'operation': {'mode': 'standalone'}
    }

@dataclass
class Config:
    # Paths (ใช้ os.path.abspath เพื่อความชัวร์ และอ่านจาก yaml ทั้งหมด)
    
    # 1. Model: แก้ Default เป็น .pt ให้ตรงกับ S3
    MODEL_PACK: str = os.path.abspath(
        yaml_cfg.get('artifacts', {}).get('model', 'models/seg_best_process.pt')
    )
    
    DB_PACKS_VEC: str = os.path.abspath(
        yaml_cfg.get('artifacts', {}).get('pack_vec', 'database/pill_fingerprints.pkl')
    )
    
    # 3. Drug List
    DRUG_LIST_JSON: str = os.path.abspath(
        yaml_cfg.get('artifacts', {}).get('drug_list', 'database/drug_list.json')
    )
    
    # Display & AI
    DISPLAY_SIZE: Tuple[int, int] = (
        yaml_cfg.get('display', {}).get('width', 1280),
        yaml_cfg.get('display', {}).get('height', 720)
    )
    AI_SIZE: int = yaml_cfg.get('settings', {}).get('ai_size', 336)
    CONF_THRESHOLD: float = yaml_cfg.get('settings', {}).get('yolo_conf', 0.5)
    MODE: str = yaml_cfg.get('operation', {}).get('mode', 'standalone')

    # Logic Constants
    VERIFY_THRESHOLD: float = 0.15
    AI_FRAME_SKIP: int = 1

    # Normalization
    MEAN: np.ndarray = field(default_factory=lambda: np.array([0.485, 0.456, 0.406], dtype=np.float32))
    STD: np.ndarray = field(default_factory=lambda: np.array([0.229, 0.224, 0.225], dtype=np.float32))

# Singleton Instance
CFG = Config()