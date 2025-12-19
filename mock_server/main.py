from fastapi import FastAPI, HTTPException
import json

app = FastAPI()

# โหลดข้อมูลจำลอง
with open("prescriptions.json", "r", encoding="utf-8") as f:
    mock_db = json.load(f)

@app.get("/get-prescription/{hn}")
async def get_prescription(hn: str):
    if hn in mock_db:
        return mock_db[hn]
    raise HTTPException(status_code=404, detail="ไม่พบข้อมูลใบสั่งยาสำหรับ HN นี้")

# วิธีรัน: uvicorn main:app --reload