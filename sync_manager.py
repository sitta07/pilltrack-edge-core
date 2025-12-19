import boto3
import os
import json
import yaml
from dotenv import load_dotenv

# --- INITIAL SETUP ---
# โหลด Environment Variables (Access Keys, Bucket Name)
load_dotenv()

# โหลดค่า Config หลักที่จะใช้กำหนด Path ของ Pack Artifacts
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_CLIENT = boto3.client('s3')

class SyncManager:
    def __init__(self):
        self.artifacts = config['artifacts']

    def get_local_timestamp(self):
        """อ่านเวลาอัปเดตล่าสุดของระบบ Pack จากไฟล์ JSON ในเครื่อง Pi"""
        path = self.artifacts.get('drug_list')
        if path and os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f).get('updated_at', '-')
            except Exception:
                return "-"
        return "-"

    def sync(self):
        """ระบบตรวจสอบและซิงค์ข้อมูลแผงยา (Pack) แบบ Mirror Path"""
        print("🔍 Scanning Pack Registry on S3 Production...")
        
        # 1. กำหนดตำแหน่ง Metadata บน S3 (เช่น latest/database/drug_list.json)
        remote_meta_path = f"latest/{self.artifacts['drug_list']}"
        local_temp_meta = "database/temp_meta.json"
        
        try:
            # ตรวจสอบการมีอยู่ของไฟล์บน Cloud ก่อนเริ่มทำงาน
            S3_CLIENT.head_object(Bucket=BUCKET_NAME, Key=remote_meta_path)
            S3_CLIENT.download_file(BUCKET_NAME, remote_meta_path, local_temp_meta)
            
            with open(local_temp_meta, 'r', encoding='utf-8') as f:
                remote_meta = json.load(f)
        except Exception as e:
            print(f"⚠️ S3 Check Failed: ไม่พบไฟล์ที่ {remote_meta_path} (Error: {e})")
            return False

        remote_ts = remote_meta.get('updated_at', '-')
        local_ts = self.get_local_timestamp()

        print(f"📍 Local Version: {local_ts}")
        print(f"📍 Cloud Version: {remote_ts}")

        # 2. เปรียบเทียบเวอร์ชันเพื่อตัดสินใจในการดาวน์โหลด
        if remote_ts != local_ts and remote_ts != "-":
            print("🚀 New Pack data detected! Starting Sync...")
            
            for key, local_path in self.artifacts.items():
                # สร้าง S3 Key โดยอิงตามโฟลเดอร์ latest
                s3_key = f"latest/{local_path}"
                
                print(f"📥 Downloading: {s3_key} -> {local_path}")
                
                # เตรียมโฟลเดอร์รองรับไฟล์ปลายทางอัตโนมัติ
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                try:
                    S3_CLIENT.download_file(BUCKET_NAME, s3_key, local_path)
                except Exception as dl_e:
                    print(f"❌ Failed to download {s3_key}: {dl_e}")
            
            print("✅ Pack identification artifacts are now up to date.")
            return True
        else:
            print("🟢 System is currently synchronized with production.")
            return False

if __name__ == "__main__":
    # ทำงานแบบอิสระเพื่อทดสอบการซิงค์
    SyncManager().sync()