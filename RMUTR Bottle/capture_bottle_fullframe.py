# capture_frames_if_bottle_every_1s.py
from pathlib import Path
import re, math, cv2, numpy as np
from ultralytics import YOLO

# =============== CONFIG ===============
VIDEO_PATH   = "MVI_3019.MP4"           # วิดีโออินพุต
OUT_DIR      = "Cropped bottle fullframe01"      # โฟลเดอร์สำหรับรูปที่เซฟ
MODEL_PATH   = "yolov8s.pt"          # โมเดลที่เทรนแล้ว (หรือ yolov8n.pt)
IMG_SIZE     = 640                            # ควรใกล้กับ imgsz ตอนเทรน (600 ≈ 608)
CONF_TH      = 0.5                           # ความมั่นใจขั้นต่ำ
INTERVAL_S   = 3                            # เก็บทุกกี่วินาที
USE_ROI      = False                          # ถ้าต้องการตรวจเฉพาะโซน ให้ True
ROI_BOX      = (200, 250, 1720, 900)          # x1,y1,x2,y2 เมื่อ USE_ROI=True
NAME_PREFIX  = "fullframe_"                       # ชื่อไฟล์เช่น fullframe_1.jpg
NAME_EXT     = ".jpg"
PAD_LETTERBOX = False                         # ไม่จำเป็น ส่วนใหญ่ปล่อย False
# ======================================

def clamp(v, lo, hi): return max(lo, min(hi, v))

def next_start_index(out_dir: Path, prefix: str, ext: str) -> int:
    pat = re.compile(rf'^{re.escape(prefix)}(\d+){re.escape(ext)}$', re.IGNORECASE)
    max_id = 0
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.is_file():
                m = pat.match(p.name)
                if m:
                    try:
                        max_id = max(max_id, int(m.group(1)))
                    except ValueError:
                        pass
    return max_id + 1 if max_id >= 1 else 1

def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    counter = next_start_index(out_dir, NAME_PREFIX, NAME_EXT)
    print(f"เริ่มนับไฟล์ที่ลำดับ: {counter}")

    model = YOLO(MODEL_PATH)
    # หา id ของคลาส 'bottle' (ต้องมีในโมเดลของคุณ)
    names = model.model.names if hasattr(model, "model") else model.names
    name_to_id = {v:k for k,v in names.items()}
    if "bottle" not in name_to_id:
        raise RuntimeError("โมเดลนี้ไม่มีคลาส 'bottle' (ตรวจชื่อคลาสใน model.names)")

    bottle_id = name_to_id["bottle"]

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"เปิดวิดีโอไม่ได้: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps): fps = 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi = None
    if USE_ROI:
        x1,y1,x2,y2 = ROI_BOX
        x1 = clamp(x1, 0, W-1); x2 = clamp(x2, 0, W-1)
        y1 = clamp(y1, 0, H-1); y2 = clamp(y2, 0, H-1)
        if x2 > x1 and y2 > y1:
            roi = (x1,y1,x2,y2)

    next_t = 0.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        t = frame_idx / fps
        frame_idx += 1

        # ตรวจเฉพาะเมื่อถึงเวลา (ทุก INTERVAL_S วินาที)
        if t + (1.0/fps) < next_t:
            continue

        view = frame
        if roi:
            x1,y1,x2,y2 = roi
            view = frame[y1:y2, x1:x2]

        # ทำนาย
        r = model.predict(
            view, imgsz=IMG_SIZE, conf=CONF_TH, verbose=False
        )[0]

        has_bottle = False
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            if np.any(cls == bottle_id):
                has_bottle = True

        if has_bottle:
            # เซฟ "ทั้งเฟรมเต็ม" (หรืออยากเซฟเฉพาะ ROI ก็เปลี่ยนเป็น view ได้)
            out_name = f"{NAME_PREFIX}{counter}{NAME_EXT}"
            cv2.imwrite(str(out_dir / out_name), frame)
            print(f"saved: {out_name}")
            counter += 1

        # ตั้งเวลาเป้าหมายรอบถัดไป
        next_t += max(0.001, INTERVAL_S)

    cap.release()
    print("done.")

if __name__ == "__main__":
    main()
