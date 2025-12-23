import cv2
import time
from src.utils.config import CFG
from src.utils.helpers import normalize_name

class UIRenderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # Colors (R, G, B, A)
        self.CYAN = (0, 255, 255, 255)
        self.YELLOW = (255, 255, 0, 255)
        self.GREEN = (0, 255, 0, 255)
        self.GRAY = (150, 150, 150, 255)
        self.RED = (0, 0, 255, 255)

    def _draw_text(self, img, text, pos, scale=0.5, color=(255,255,255,255), thickness=1):
        black = (0, 0, 0, 255)
        cv2.putText(img, text, pos, self.font, scale, black, thickness+2)
        cv2.putText(img, text, pos, self.font, scale, color, thickness)

    def draw(self, frame, ai_processor):
        """Main Draw Function that takes AI state and renders UI"""
        rx = ai_processor.rx
        
        # 1. Status Bar
        status_color = self.GREEN if rx.is_completed else self.CYAN
        status_text = "COMPLETED - RESETTING..." if rx.is_completed else f"PATIENT: {rx.patient_name}"
        self._draw_text(frame, status_text, (20, CFG.DISPLAY_SIZE[1] - 30), 0.7, status_color, 2)

        # 2. Prescription List
        y_pos = 50
        self._draw_text(frame, "PRESCRIPTION:", (CFG.DISPLAY_SIZE[0] - 250, 30), 0.6, self.YELLOW, 2)
        
        for norm, data in rx.target_drugs.items():
            is_found = data['found'] > 0
            color = self.GREEN if is_found else self.GRAY
            icon = "✔" if is_found else "□"
            text = f"{icon} {data['original'].upper()} x{data['qty']}"
            self._draw_text(frame, text, (CFG.DISPLAY_SIZE[0] - 240, y_pos), 0.5, color, 1)
            y_pos += 30

        # 3. Detection Boxes
        if not rx.is_completed:
            with ai_processor.lock:
                for res in ai_processor.results:
                    x1, y1, x2, y2 = res['box']
                    color = self.GREEN if (res['is_correct'] and res['conf'] > 0.8) else (self.CYAN if res['is_correct'] else self.RED)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label_text = f"{res['label']} {res['conf']:.2f}"
                    if not res['is_correct']: label_text = f"WRONG: {res['label']}"
                    self._draw_text(frame, label_text, (x1, y1-5), 0.4, color, 1)

        # 4. Timer UI
        timer_pos = (CFG.DISPLAY_SIZE[0] - 350, CFG.DISPLAY_SIZE[1] - 30)
        if ai_processor.timer_running:
            elapsed = time.time() - ai_processor.timer_start_time
            self._draw_text(frame, f"⏱️ TIME: {elapsed:.2f} s", timer_pos, 0.8, self.YELLOW, 2)
        elif ai_processor.timer_result_text:
            self._draw_text(frame, f"🏁 {ai_processor.timer_result_text}", timer_pos, 0.7, self.GREEN, 2)
        else:
            self._draw_text(frame, "[T] Timer | [N] Next | [Q] Quit", timer_pos, 0.5, self.GRAY, 1)

        return frame