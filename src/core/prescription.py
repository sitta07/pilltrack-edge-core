import os
import json
import time
from typing import Dict
from src.utils.config import CFG
from src.utils.helpers import normalize_name

class PrescriptionManager:
    def __init__(self):
        self.target_drugs = {} 
        self.patient_name = "Standalone"
        self.is_ready = False
        self.is_completed = False
        self.complete_timestamp = 0
        
        if CFG.MODE == "standalone":
            self.load_local_all()
            self.is_ready = True

    def load_local_all(self):
        if not os.path.exists(CFG.DRUG_LIST_JSON): return
        try:
            with open(CFG.DRUG_LIST_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for d in data.get('drugs', []):
                self.target_drugs[normalize_name(d)] = {"original": d, "qty": None, "found": 0}
        except Exception as e:
            print(f"⚠️ Error loading drug list: {e}")

    def update_from_his(self, his_data: Dict):
        self.reset()
        self.target_drugs = {}
        self.patient_name = his_data.get('patient_name', 'Unknown')
        
        for item in his_data.get('prescription', []):
            norm = normalize_name(item['name'])
            self.target_drugs[norm] = {
                "original": item['name'],
                "qty": item['amount'],
                "found": 0
            }
        self.is_ready = True
        print(f"📦 Prescription Loaded: {self.patient_name}")

    def verify(self, detected_name: str):
        if self.is_completed: return False
        norm_det = normalize_name(detected_name)
        
        if norm_det in self.target_drugs:
            self.target_drugs[norm_det]['found'] = 1
            self.check_complete()
            return True
        return False

    def check_complete(self):
        if not self.target_drugs: return
        all_found = all(d['found'] > 0 for d in self.target_drugs.values())
        if all_found and not self.is_completed:
            self.is_completed = True
            self.complete_timestamp = time.time()
            print(f"✅ COMPLETED FOR {self.patient_name}!")

    def reset(self):
        self.target_drugs = {}
        self.patient_name = "Waiting..."
        self.is_ready = False
        self.is_completed = False
        self.complete_timestamp = 0