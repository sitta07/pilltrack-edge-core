import boto3
import os
import yaml
from datetime import datetime, timezone
from botocore.exceptions import ClientError, NoCredentialsError

class SyncManager:
    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("❌ S3_BUCKET_NAME is missing in .env file")

        self.s3 = boto3.client('s3')
        
        self.files_to_sync = {
            "latest/register_model/pill_fingerprints.pkl": cfg['artifacts']['pack_vec'],
            "latest/register_model/drug_list.json": cfg['artifacts']['drug_list'],
            "latest/models/seg_best_process.pt": cfg['artifacts']['model'] # ✅ แก้ให้ตรงชื่อไฟล์จริง
        }

    def _is_s3_newer(self, s3_key, local_path):
        if not os.path.exists(local_path):
            return True 

        try:
            obj = self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
            s3_time = obj['LastModified']
            
            local_timestamp = os.path.getmtime(local_path)
            local_time = datetime.fromtimestamp(local_timestamp, tz=timezone.utc)

            return s3_time > local_time
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                print(f"⚠️  File not found on S3: {s3_key}")
            return False

    def sync(self):
        print(f"☁️  Checking bucket: '{self.bucket_name}'...")
        
        # Test Connection
        self.s3.head_bucket(Bucket=self.bucket_name)

        updates_count = 0
        for s3_key, local_path in self.files_to_sync.items():
            try:
                if self._is_s3_newer(s3_key, local_path):
                    print(f"⬇️  Downloading: {s3_key} -> {local_path}")
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    self.s3.download_file(self.bucket_name, s3_key, local_path)
                    updates_count += 1
            except Exception as e:
                print(f"❌ Failed to download {s3_key}: {e}")

        if updates_count > 0:
            print(f"✨ Update Complete! ({updates_count} files updated)")
        else:
            print("✨ System is up to date.")

if __name__ == "__main__":
    # Test Run เฉพาะไฟล์นี้
    from dotenv import load_dotenv
    load_dotenv()
    try:
        SyncManager().sync()
    except Exception as e:
        print(e)