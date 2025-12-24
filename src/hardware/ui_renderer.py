import cv2
import time
from src.utils.config import CFG
# from src.utils.helpers import normalize_name # ถ้าไม่ได้ใช้ก็ comment ไว้ก่อนกัน error

class UIRenderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # ✅ แก้จุดที่ 1 & 2: เปลี่ยนเป็น RGB (3 ตัว) ให้ตรงกับกล้อง
        # Format: (Red, Green, Blue)
        self.CYAN = (0, 255, 255)    # R=0, G=255, B=255
        self.YELLOW = (255, 255, 0)  # R=255, G=255, B=0
        self.GREEN = (0, 255, 0)     # R=0, G=255, B=0
        self.GRAY = (150, 150, 150)
        self.RED = (255, 0, 0)       # R=255, G=0, B=0 (แก้ตรงนี้สำคัญสุด!)
        self.BLACK = (0, 0, 0)

    def _draw_text(self, img, text, pos, scale=0.5, color=(255,255,255), thickness=1):
        # วาดขอบดำก่อน (Outline) เพื่อให้อ่านง่าย
        cv2.putText(img, text, pos, self.font, scale, self.BLACK, thickness+2)
        # วาดตัวหนังสือสีทับลงไป
        cv2.putText(img, text, pos, self.font, scale, color, thickness)

    def draw(self, frame, ai_processor):
        """Main Draw Function"""
        rx = ai_processor.rx
        
        # 1. Status Bar
        status_color = self.GREEN if rx.is_completed else self.CYAN
        status_text = "COMPLETED - RESETTING..." if rx.is_completed else f"PATIENT: {rx.patient_name}"
        self._draw_text(frame, status_text, (20, CFG.DISPLAY_SIZE[1] - 30), 0.7, status_color, 2)

        # 2. Prescription List
        y_pos = 50
        self._draw_text(frame, "PRESCRIPTION:", (CFG.DISPLAY_SIZE[0] - 250, 30), 0.6, self.YELLOW, 2)
        
        # ใช้ .get() เผื่อ key ไม่มี จะได้ไม่ crash
        target_drugs = getattr(rx, 'target_drugs', {}) 
        
        for norm, data in target_drugs.items():
            is_found = data['found'] > 0
            color = self.GREEN if is_found else self.GRAY
            icon = "v" if is_found else "[ ]"  # เปลี่ยน icon เป็นตัวอักษรธรรมดา กัน Font ใน Linux ไม่มี emoji
            
            # ใช้ .get() ดึงค่าเผื่อ data ไม่ครบ
            qty = data.get('qty', 0)
            orig_name = data.get('original', 'Unknown')
            
            text = f"{icon} {orig_name.upper()} x{qty}"
            self._draw_text(frame, text, (CFG.DISPLAY_SIZE[0] - 240, y_pos), 0.5, color, 1)
            y_pos += 30

        # 3. Detection Boxes
        if not rx.is_completed:
            # ใช้ try-except กัน error ตอนวาดกล่อง
            try:
                # เช็กว่า results มีของไหม
                if hasattr(ai_processor, 'results'):
                    for res in ai_processor.results:
                        x1, y1, x2, y2 = map(int, res['box']) # แปลงเป็น int ให้ชัวร์
                        
                        is_correct = res.get('is_correct', False)
                        conf = res.get('conf', 0.0)
                        label = res.get('label', '?')

                        color = self.GREEN if (is_correct and conf > 0.8) else (self.CYAN if is_correct else self.RED)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label_text = f"{label} {conf:.2f}"
                        if not is_correct: label_text = f"WRONG: {label}"
                        self._draw_text(frame, label_text, (x1, y1-5), 0.4, color, 1)
            except Exception as e:
                print(f"Draw Error: {e}")

        # 4. Timer UI
        timer_pos = (CFG.DISPLAY_SIZE[0] - 350, CFG.DISPLAY_SIZE[1] - 30)
        
        if getattr(ai_processor, 'timer_running', False):
            elapsed = time.time() - ai_processor.timer_start_time
            self._draw_text(frame, f"TIME: {elapsed:.2f} s", timer_pos, 0.8, self.YELLOW, 2)
        elif getattr(ai_processor, 'timer_result_text', ""):
            self._draw_text(frame, f"FINISH: {ai_processor.timer_result_text}", timer_pos, 0.7, self.GREEN, 2)
        else:
            self._draw_text(frame, "[T] Timer | [N] Next", timer_pos, 0.5, self.GRAY, 1)

        return frame