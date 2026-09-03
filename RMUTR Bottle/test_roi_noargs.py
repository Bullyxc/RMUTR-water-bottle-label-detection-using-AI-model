# detect_after_resize_then_roi.py
from ultralytics import YOLO
import cv2

# ===================== CONFIG =====================
MODEL_PATH  = "runs/detect/train/weights/best.pt"
VIDEO_PATH  = "MVI_3018.MP4"

# ปรับให้พอดีกับจอ: ใส่ "กรอบมากสุด" ที่อยากให้โชว์ (จะคงอัตราส่วนเดิมไว้)
DISPLAY_MAX_W = 1280
DISPLAY_MAX_H = 720

# กำหนด ROI บน "ภาพที่ย่อแล้ว" (หน่วยพิกเซลของภาพหลังย่อ)
ROI_DISP = (350, 160, 1200, 420)   # x1, y1, x2, y2  <<< ปรับตรงนี้ให้เหมาะกับจอคุณ

IMG_SIZE  = 600     # ควรใกล้กับขนาดที่เทรน (คุณใช้ 600 ≈ 608)
CONF_TH   = 0.5
IOU_TH    = 0.45
SHOW_FULL = True    # โชว์ภาพเต็มหลังย่อ + วาดกล่อง (นอกจาก ROI)
# ===================================================

def clamp(v, lo, hi): return max(lo, min(hi, v))

m = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("เปิดวิดีโอไม่ได้")

while True:
    ok, f = cap.read()
    if not ok: break

    H, W = f.shape[:2]
    scale = min(DISPLAY_MAX_W / W, DISPLAY_MAX_H / H)  # คงสัดส่วน
    newW, newH = int(W * scale), int(H * scale)
    disp = cv2.resize(f, (newW, newH), interpolation=cv2.INTER_AREA)

    # ครอป ROI จาก "ภาพที่ย่อแล้ว"
    x1,y1,x2,y2 = ROI_DISP
    x1 = clamp(x1, 0, newW-1); x2 = clamp(x2, 0, newW-1)
    y1 = clamp(y1, 0, newH-1); y2 = clamp(y2, 0, newH-1)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI_DISP ไม่ถูกต้อง (x2<=x1 หรือ y2<=y1)")

    roi = disp[y1:y2, x1:x2]

    # รันตรวจจับบน ROI
    r = m.predict(roi, imgsz=IMG_SIZE, conf=CONF_TH, iou=IOU_TH, verbose=False)[0]
    vis_roi = r.plot()

    # แสดงผล
    cv2.imshow("ROI (after resize)", vis_roi)

    if SHOW_FULL:
        vis_full = disp.copy()
        cv2.rectangle(vis_full, (x1,y1), (x2,y2), (0,255,255), 2)
        cv2.imshow("Full (resized) + ROI box", vis_full)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

cap.release(); cv2.destroyAllWindows()
