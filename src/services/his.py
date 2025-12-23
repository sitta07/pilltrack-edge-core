import json
import requests
import os
from src.utils.config import CFG

class HISConnector:
    def __init__(self):
        self.mode = CFG.MODE 
        self.mock_db_path = "mock_server/prescriptions.json"
        
        self.mock_data = {}
        if self.mode == 'standalone':
            self._load_mock_db()

    def _load_mock_db(self):
        if os.path.exists(self.mock_db_path):
            with open(self.mock_db_path, 'r', encoding='utf-8') as f:
                self.mock_data = json.load(f)
            print(f"🏥 HIS (Mock): Loaded {len(self.mock_data)} patients.")
        else:
            print("⚠️ HIS (Mock): Database file not found!")

    def fetch_prescription(self, hn_number: str):
        """
        ฟังก์ชันหลักที่ Main เรียกใช้
        """
        print(f"📡 Fetching data for HN: {hn_number}...")

        # CASE 1: ใช้ Mock Data (แนะนำให้ใช้ตอนนี้)
        if self.mode == 'standalone':
            data = self.mock_data.get(hn_number)
            if data:
                print(f"✅ Found (Mock): {data['patient_name']}")
                return data
            else:
                print("❌ HN not found in mock DB.")
                return None

        # CASE 2: ยิง API จริง (เผื่อไว้ในอนาคต)
        # elif self.mode == 'connected':
        #     try:
        #         # ตัวอย่างการยิง API
        #         # res = requests.get(f"http://api.hospital.com/rx/{hn_number}", timeout=5)
        #         # return res.json()
        #         pass
        #     except Exception as e:
        #         print(f"❌ Network Error: {e}")
        #         return None

        return None