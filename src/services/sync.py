import boto3
import os
import yaml
from datetime import datetime, timezone
from botocore.exceptions import ClientError, NoCredentialsError

class SyncManager:
    def __init__(self, config_path="config.yaml"):
        self.s3 = boto3.client('s3')
        
        # Load Config เพื่อหาชื่อ Bucket
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            
        # ดึงค่าจาก env หรือ config.yaml
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'your-bucket-name')
        
        # กำหนดคู่ไฟล์ (S3 Path -> Local Path)
        # S3 Path: ตำแหน่งไฟล์บน Cloud (เช่น 'latest/register_model/model.pt')
        # Local Path: ที่ที่จะเซฟลง Pi (เช่น 'models/model.pt')
        self.files_to_sync = {
            "latest/register_model/pill_fingerprints.pkl": cfg['artifacts']['pack_vec'],
            "latest/register_model/drug_list.json": cfg['artifacts']['drug_list'],
            "latest/models/seg_best_process.pt": cfg['artifacts']['model']
        }

    def _is_s3_newer(self, s3_key, local_path):
        """เช็คว่าไฟล์บน S3 ใหม่กว่าในเครื่องไหม"""
        if not os.path.exists(local_path):
            return True # ถ้าไม่มีไฟล์ในเครื่อง โหลดใหม่แน่นอน

        try:
            # ดึง Metadata ของไฟล์บน S3
            obj = self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            s3_time = obj['LastModified']
            
            # ดึงเวลาไฟล์ในเครื่อง (Convert เป็น Timezone เดียวกัน)
            local_timestamp = os.path.getmtime(local_path)
            local_time = datetime.fromtimestamp(local_timestamp, tz=timezone.utc)

            # เทียบเวลา (ถ้า S3 ใหม่กว่า -> True)
            return s3_time > local_time
        except ClientError:
            print(f"⚠️ S3 File not found: {s3_key}")
            return False

    def sync(self):
        print("☁️ Checking for updates from Cloud...")
        
        try:
            # เช็คเน็ตก่อนเบื้องต้น
            self.s3.head_bucket(Bucket=self.bucket_name)
        except (ClientError, NoCredentialsError):
            print("❌ Offline mode or No Credentials. Skipping Sync.")
            return

        updates_count = 0
        for s3_key, local_path in self.files_to_sync.items():
            try:
                if self._is_s3_newer(s3_key, local_path):
                    print(f"⬇️ Downloading update: {s3_key} -> {local_path}")
                    
                    # สร้าง Folder รอถ้ายังไม่มี
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    self.s3.download_file(self.bucket_name, s3_key, local_path)
                    updates_count += 1
                else:
                    pass
                    # print(f"✅ Up to date: {local_path}") # Uncomment ถ้าอยากเห็น Log
            except Exception as e:
                print(f"❌ Failed to sync {s3_key}: {e}")

        if updates_count > 0:
            print(f"✨ Update Complete! ({updates_count} files updated)")
        else:
            print("✨ System is up to date.")

if __name__ == "__main__":
    # Test Run
    SyncManager().sync()