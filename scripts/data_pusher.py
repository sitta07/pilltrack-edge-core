import os
import shutil
import boto3
import yaml
from datetime import datetime
from dotenv import load_dotenv

# ================= ⚙️ SETUP =================
load_dotenv()

def load_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f) or {}
    return {}

config = load_config()

# ดึงค่าจาก Config และ Env
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
# โฟลเดอร์ที่เก็บภาพที่ Pi ถ่ายไว้
SOURCE_DIR = "captured_images" 
# ปลายทางบน S3 (เก็บแยกไว้ใน folder 'data_collection')
S3_PREFIX = "data_collection" 

s3_client = boto3.client('s3')

def push_captured_data():
    """ระบบบีบอัดและส่งข้อมูลกลับ Cloud [cite: 2025-11-11, 2025-12-05]"""
    
    # 1. เช็คว่ามีรูปให้ส่งไหม
    if not os.path.exists(SOURCE_DIR) or not os.listdir(SOURCE_DIR):
        print(f"📭 No images found in {SOURCE_DIR}. Skipping...")
        return False

    # 2. เตรียมชื่อไฟล์ด้วย Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"captured_{timestamp}" # ไม่ต้องใส่ .zip เดี๋ยว shutil ใส่ให้
    zip_path = f"{zip_filename}.zip"

    try:
        print(f"📦 Zipping {SOURCE_DIR}...")
        # บีบอัดโฟลเดอร์เป็น .zip [cite: 2025-11-11]
        shutil.make_archive(zip_filename, 'zip', SOURCE_DIR)

        # 3. อัปโหลดขึ้น S3
        s3_key = f"{S3_PREFIX}/{zip_path}"
        print(f"🚀 Pushing to S3: s3://{BUCKET_NAME}/{s3_key}...")
        
        s3_client.upload_file(zip_path, BUCKET_NAME, s3_key)
        print("✅ Upload Successful!")

        # 4. Cleanup (สไตล์ Senior: ทำงานเสร็จต้องล้างบ้าน) [cite: 2025-12-05]
        print("🧹 Cleaning up local data...")
        # ลบไฟล์ Zip
        if os.path.exists(zip_path):
            os.remove(zip_path)
        
        # ลบรูปภาพต้นฉบับในโฟลเดอร์ทิ้ง (เพื่อให้ Pi ว่างสำหรับรอบถัดไป)
        for filename in os.listdir(SOURCE_DIR):
            file_path = os.path.join(SOURCE_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        print("✨ Done! Pi is ready for new captures.")
        return True

    except Exception as e:
        print(f"❌ Failed to push data: {str(e)}")
        # หากส่งไม่สำเร็จ จะไม่ลบรูปทิ้ง เผื่อส่งใหม่คราวหน้า [cite: 2025-11-11]
        return False

if __name__ == "__main__":
    push_captured_data()