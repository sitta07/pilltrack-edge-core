import requests
import yaml

class HISConnector:
    def __init__(self, config_path="config.yaml"):
        # โหลด Config เพื่อเอา URL
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.base_url = config['his_api']['base_url']
        self.timeout = config['his_api']['timeout']
        self.endpoint = config['his_api']['endpoints']['get_prescription']

    def fetch_prescription(self, hn: str):
        """
        ดึงข้อมูลใบสั่งยาจาก HIS ด้วย HN (Hospital Number)
        """
        try:
            url = f"{self.base_url}{self.endpoint}{hn}"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ ดึงข้อมูลสำเร็จ: {data['patient_name']}")
                return data
            elif response.status_code == 404:
                print(f"❌ ไม่พบข้อมูล HN: {hn}")
                return None
            else:
                print(f"⚠️ Server Error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"🚨 เชื่อมต่อ HIS ไม่ได้: {e}")
            return None

# เอาไว้ Test แยก (Standalone test)
if __name__ == "__main__":
    connector = HISConnector()
    res = connector.fetch_prescription("HN123")
    print(res)