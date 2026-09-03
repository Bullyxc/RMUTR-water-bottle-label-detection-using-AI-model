
from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG — แก้ค่าตรงนี้ได้เลย
# =========================
MODEL_PATH      = "runs2/detect/train/weights/best.pt"   # .pt ที่คุณเทรนเสร็จ
VIDEO_PATH      = "MVI_3019.MP4"      # วิดีโอทดสอบ
IMG_SIZE        = 640                           # ขนาดอินพุตโมเดล (ขวดเล็กมากลอง 960/1024)
CONF_TH         = 0.5                           # ค่าความมั่นใจขั้นต่ำ
IOU_TH          = 0.6                           # NMS IoU threshold 0.45 default
DEVICE          = "cpu"                              # 0=GPU ตัวแรก, "cpu"=ใช้ CPU
SHOW_WINDOW     = True                           # แสดงผลบนหน้าต่างหรือไม่
DISPLAY_RESIZE  = 0.8                           # สเกลสำหรับหน้าต่าง (เช่น 0.7 ลดขนาดโชว์)
WRITE_VIDEO     = False                           # บันทึกวิดีโอผลลัพธ์หรือไม่
OUTPUT_PATH     = "RMUTR Bottle"   # ที่เก็บวิดีโอผลลัพธ์
DRAW_ONLY_CLASSES = []
# DRAW_ONLY_CLASSES = ["bottle", "label"]          # วาดเฉพาะคลาสเหล่านี้ (หรือ [] = วาดทุกคลาส)
THICKNESS       = 2                              # ความหนาของเส้นกรอบ
FONT_SCALE      = 0.6                            # ขนาดฟอนต์ข้อความ
# =========================

def put_text(img, text, org, font_scale=0.6, color=(0,255,0), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def main():
    # โหลดโมเดล
    model = YOLO(MODEL_PATH)

    # mapping class id -> name
    names = model.model.names if hasattr(model, "model") else model.names
    # เตรียมชุดคลาสที่ต้องการวาด (แปลงเป็น id ถ้าระบุชื่อมา)
    wanted_ids = None
    if DRAW_ONLY_CLASSES:
        name_to_id = {v: k for k, v in names.items()}  # id->name -> name->id
        wanted_ids = set([name_to_id[c] for c in DRAW_ONLY_CLASSES if c in name_to_id])

    # เปิดวิดีโอ
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"เปิดวิดีโอไม่สำเร็จ: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ตั้ง VideoWriter ถ้าต้องการบันทึก
    writer = None
    if WRITE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_size = (W, H)
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, out_size)

    prev_time = time.time()
    stem = Path(VIDEO_PATH).stem
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # รันตรวจจับ
        results = model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_TH,
            iou=IOU_TH,
            device=DEVICE,
            verbose=False
        )

        # วาดผล
        vis = frame.copy()
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
                # กรองคลาส ถ้ากำหนด
                if wanted_ids is not None and c not in wanted_ids:
                    continue
                label = names.get(c, f"id{c}")
                txt = f"{label} {p:.2f}"
                # วาดกรอบ + ป้าย
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), THICKNESS)
                put_text(vis, txt, (x1, max(20, y1-8)), FONT_SCALE, (0, 200, 255), 2)

        # คำนวณ FPS แสดงมุมจอ
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps_est = 1.0 / dt if dt > 0 else 0.0
        put_text(vis, f"{stem} | frame {frame_idx} | FPS {fps_est:.1f}", (10, 28), 0.7, (50,255,50), 2)

        # แสดงหน้าต่าง
        if SHOW_WINDOW:
            if DISPLAY_RESIZE != 1.0:
                vis_show = cv2.resize(vis, (int(W*DISPLAY_RESIZE), int(H*DISPLAY_RESIZE)))
            else:
                vis_show = vis
            cv2.imshow("YOLOv8 Detection", vis_show)
            # กด Q เพื่อออก
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break

        # เขียนวิดีโอผลลัพธ์
        if writer is not None:
            writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()
        print(f"บันทึกผลที่: {OUTPUT_PATH}")
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print("เสร็จสิ้น.")

if __name__ == "__main__":
    main()
