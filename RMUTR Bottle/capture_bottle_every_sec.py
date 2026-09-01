from pathlib import Path
import cv2
import math
import numpy as np
import re
from ultralytics import YOLO

# =========================
# CONFIG — แก้ค่าตรงนี้ได้เลย
# =========================
VIDEO_PATH = "MVI_3018.MP4"            # วิดีโออินพุต
OUT_DIR    = "Cropped bottle03"       # โฟลเดอร์สำหรับเซฟรูปครอป
INTERVAL_S = 3                            # เก็บทุกกี่วินาที (เช่น 1.0 = ทุก 1 วินาที)
CONF_TH    = 0.5                          # ความมั่นใจขั้นต่ำว่าคือขวด
PAD_RATIO  = 0.08                           # เผื่อขอบรอบกรอบครอป (สัดส่วนต่อ w/h กรอบ)
IMG_SIZE   = 640                            # ขนาดอินพุต YOLO (ขวดเล็กมากลอง 960/1024)
USE_ROI    = False                           # ใช้ ROI เฉพาะโซนสายพานหรือไม่
ROI_BOX    = (200, 250, 1720, 900)          # x1,y1,x2,y2 (พิกเซล) ถ้า USE_ROI=False จะไม่ใช้ค่านี้
SAVE_DEBUG_FRAME = False                    # เซฟรูปเต็มเฟรมพร้อมกล่องตรวจจับเพื่อเช็คผล
MODEL_PATH = "yolov8s.pt"                   # โมเดล COCO pretrained (มีคลาส bottle)
# การตั้งชื่อไฟล์แบบลำดับ
NAME_PREFIX = "bottleV2_"                      # คำนำหน้าไฟล์
NAME_EXT    = ".jpg"                         # นามสกุลไฟล์
# =========================

def clamp(v, lo, hi): return max(lo, min(hi, v))

def next_start_index(out_dir: Path, prefix: str, ext: str) -> int:
    """
    หาเลขลำดับถัดไป โดยดูไฟล์ที่มีรูปแบบ: {prefix}{number}{ext}
    เช่น bottle_123.jpg -> จะเริ่มที่ 124
    """
    pat = re.compile(rf'^{re.escape(prefix)}(\d+){re.escape(ext)}$', re.IGNORECASE)
    max_id = 0
    if not out_dir.exists():
        return 1
    for p in out_dir.iterdir():
        if not p.is_file():
            continue
        m = pat.match(p.name)
        if m:
            try:
                n = int(m.group(1))
                if n > max_id:
                    max_id = n
            except ValueError:
                pass
    return max_id + 1 if max_id >= 1 else 1

def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # เตรียมตัวนับลำดับต่อจากไฟล์เดิม
    counter = next_start_index(out_dir, NAME_PREFIX, NAME_EXT)
    print(f"เริ่มนับไฟล์ที่ลำดับ: {counter}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"เปิดวิดีโอไม่สำเร็จ: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps): fps = 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi_box = None
    if USE_ROI:
        x1,y1,x2,y2 = ROI_BOX
        x1 = clamp(x1, 0, W-1); x2 = clamp(x2, 0, W-1)
        y1 = clamp(y1, 0, H-1); y2 = clamp(y2, 0, H-1)
        if x2 > x1 and y2 > y1:
            roi_box = (x1,y1,x2,y2)

    model = YOLO(MODEL_PATH)

    # หา class id ของ "bottle"
    names = model.model.names if hasattr(model, "model") else model.names
    name_to_id = {v:k for k,v in names.items()}
    if "bottle" not in name_to_id:
        raise RuntimeError("โมเดลนี้ไม่มีคลาส 'bottle'")
    bottle_id = name_to_id["bottle"]

    next_t = 0.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        t = frame_idx / fps
        frame_idx += 1

        if t + (1.0/fps) < next_t:
            continue  # ยังไม่ถึงเวลาตรวจ

        view = frame
        xoff = yoff = 0
        if roi_box:
            x1,y1,x2,y2 = roi_box
            view = frame[y1:y2, x1:x2]
            xoff, yoff = x1, y1

        res = model.predict(view, imgsz=IMG_SIZE, conf=CONF_TH, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes
            cls = boxes.cls.cpu().numpy().astype(int)
            conf = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy().astype(int)

            inds = np.where(cls == bottle_id)[0]
            if len(inds) > 0:
                bi = inds[np.argmax(conf[inds])]
                x1, y1, x2, y2 = xyxy[bi]

                bw, bh = x2 - x1, y2 - y1
                px, py = int(bw * PAD_RATIO), int(bh * PAD_RATIO)
                x1p = clamp(x1 - px, 0, view.shape[1]-1)
                y1p = clamp(y1 - py, 0, view.shape[0]-1)
                x2p = clamp(x2 + px, 0, view.shape[1]-1)
                y2p = clamp(y2 + py, 0, view.shape[0]-1)

                X1, Y1 = x1p + xoff, y1p + yoff
                X2, Y2 = x2p + xoff, y2p + yoff
                crop = frame[Y1:Y2, X1:X2]

                # ตั้งชื่อไฟล์แบบเลขลำดับต่อเนื่อง
                out_name = f"{NAME_PREFIX}{counter}{NAME_EXT}"
                cv2.imwrite(str(out_dir / out_name), crop)
                print(f"saved: {out_name}")
                counter += 1

                # (อ็อปชัน) เซฟ debug frame (ไม่ใช้นับลำดับ)
                if SAVE_DEBUG_FRAME:
                    dbg = frame.copy()
                    cv2.rectangle(dbg, (X1, Y1), (X2, Y2), (0,255,0), 2)
                    cv2.putText(dbg, f"t={t:.2f}s", (X1, max(0, Y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.imwrite(str(out_dir / f"DEBUG_t{t:07.2f}_f{frame_idx-1}.jpg"), dbg)

        next_t += max(0.001, INTERVAL_S)

    cap.release()
    print("done.")

if __name__ == "__main__":
    main()
