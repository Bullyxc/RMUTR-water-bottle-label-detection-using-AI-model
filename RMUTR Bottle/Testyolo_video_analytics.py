#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLOv8 Video Test + Analytics (Fixed paths, no CLI)
- แก้ค่าที่บล็อก CONFIG แล้วรันได้เลย
- บันทึกวิดีโอใส่กรอบ + กราฟ + CSV + report.md

ต้องมี:
pip install ultralytics opencv-python matplotlib pandas numpy
"""

import os, time, sys
from collections import defaultdict, Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# =========================
# CONFIG (แก้ตรงนี้ให้ตรง env)
# =========================
MODEL_PATH = "runs2/detect/train/weights/best.pt"   # path โมเดล .pt
VIDEO_PATH = "MVI_3018.MP4"                    # path วิดีโอที่มีอยู่ในเครื่อง
SAVE_DIR   = "Testresult01"                   # โฟลเดอร์ผลลัพธ์
IMGSZ      = 640                                    # ขนาดอินพุตของโมเดล
CONF       = 0.5                                   # confidence threshold
IOU        = 0.45                                   # NMS IoU threshold
DEVICE     = None                                   # "cuda:0" หรือ "cpu" (None = auto)
AUTO_OPEN_FOLDER = False                            # True = เปิดโฟลเดอร์ผลลัพธ์เมื่อจบ

# =========================
# Utilities
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def draw_text(img, text, x, y, scale=0.6, color=(255,255,255), thickness=1):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

# =========================
# Main
# =========================
def main():
    # ตรวจพาธพื้นฐาน
    if not os.path.isfile(MODEL_PATH):
        print(f"❌ ไม่พบโมเดล: {MODEL_PATH}"); sys.exit(1)
    if not os.path.isfile(VIDEO_PATH):
        print(f"❌ ไม่พบวิดีโอ: {VIDEO_PATH}"); sys.exit(1)

    # โฟลเดอร์ผลลัพธ์ (แยกตามเวลา เพื่อไม่ทับของเก่า)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join(SAVE_DIR, f"run_{stamp}")
    frames_dir = ensure_dir(os.path.join(save_root, "frames"))
    plots_dir  = ensure_dir(os.path.join(save_root, "plots"))
    csv_dir    = ensure_dir(os.path.join(save_root, "csv"))

    print("▶ โหลดโมเดล:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    print("▶ เปิดวิดีโอ:", VIDEO_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ เปิดวิดีโอไม่ได้: {VIDEO_PATH}")
        sys.exit(1)

    in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    out_video_path = os.path.join(save_root, "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps_src, (in_w, in_h))

    # เก็บสถิติ
    t0 = time.time()
    frame_idx = 0
    proc_times, det_counts = [], []
    conf_all, box_area_rel = [], []
    conf_by_class = defaultdict(list)
    class_count = Counter()
    rows_csv = []

    print("▶ เริ่มประมวลผล…")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        t1 = time.time()
        results = model.predict(
            source=frame, imgsz=IMGSZ, conf=CONF, iou=IOU, device=DEVICE, verbose=False
        )
        dt = time.time() - t1
        proc_times.append(dt)

        res = results[0]
        names = res.names
        boxes = res.boxes
        n_det = 0

        plotted = res.plot()

        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                n_det += 1
                cls_id = int(b.cls.item())
                c = float(b.conf.item()) if b.conf is not None else None
                x1,y1,x2,y2 = b.xyxy.cpu().numpy().reshape(-1)
                w, h = max(0.0,x2-x1), max(0.0,y2-y1)
                area_rel = (w*h)/(in_w*in_h + 1e-9)
                class_count[cls_id] += 1
                if c is not None:
                    conf_all.append(c); conf_by_class[cls_id].append(c)
                box_area_rel.append(area_rel)

        det_counts.append(n_det)

        cur_fps = 1.0/dt if dt>0 else 0.0
        draw_text(plotted, f"Frame: {frame_idx}", 10, 25)
        draw_text(plotted, f"Detections: {n_det}", 10, 50)
        draw_text(plotted, f"FPS(proc): {cur_fps:.1f}", 10, 75)

        if n_det>0:
            names_in_frame = [names.get(int(b.cls.item()), str(int(b.cls.item()))) for b in boxes]
            draw_text(plotted, "Classes: " + ", ".join(names_in_frame[:6]), 10, 100)

        writer.write(plotted)

        rows_csv.append({"frame": frame_idx, "proc_time_s": dt, "fps_proc": cur_fps, "n_det": n_det})

        if total_est and frame_idx % 100 == 0:
            print(f"  ... {frame_idx}/{total_est}")

    cap.release(); writer.release()
    total_time = time.time() - t0

    # บันทึก CSV
    df = pd.DataFrame(rows_csv)
    df.to_csv(os.path.join(csv_dir, "per_frame_stats.csv"), index=False)

    # วาดกราฟ (อย่าลืมปิด figure ทุกครั้ง)
    def save_plot(figpath):
        plt.tight_layout(); plt.savefig(figpath); plt.close()

    # 1) detections over time
    if len(det_counts)>0:
        plt.figure(); plt.plot(range(1, len(det_counts)+1), det_counts)
        plt.xlabel("Frame"); plt.ylabel("Detections per frame"); plt.title("Detections over time")
        save_plot(os.path.join(plots_dir, "detections_over_time.png"))

    # 2) confidence hist
    if len(conf_all)>0:
        plt.figure(); plt.hist(conf_all, bins=20, range=(0,1))
        plt.xlabel("Confidence"); plt.ylabel("Count"); plt.title("Confidence distribution (all classes)")
        save_plot(os.path.join(plots_dir, "confidence_hist.png"))

    # 3) class counts
    if len(class_count)>0:
        names_map = model.names if hasattr(model, "names") else {k:str(k) for k in class_count.keys()}
        cls_ids, counts = zip(*sorted(class_count.items(), key=lambda x:-x[1]))
        labels = [names_map.get(i, str(i)) for i in cls_ids]
        plt.figure(); plt.bar(labels, counts); plt.xticks(rotation=45, ha="right")
        plt.ylabel("Total detections"); plt.title("Detections by class")
        save_plot(os.path.join(plots_dir, "class_counts.png"))

    # 4) bbox area hist
    if len(box_area_rel)>0:
        plt.figure(); plt.hist(box_area_rel, bins=30)
        plt.xlabel("Box area / Frame area"); plt.ylabel("Count"); plt.title("Bounding box relative area distribution")
        save_plot(os.path.join(plots_dir, "box_area_hist.png"))

    # 5) FPS over time
    if len(proc_times)>0:
        fps_series = [1.0/max(t,1e-9) for t in proc_times]
        plt.figure(); plt.plot(range(1, len(fps_series)+1), fps_series)
        plt.xlabel("Frame"); plt.ylabel("Processing FPS"); plt.title("Processing FPS over time")
        save_plot(os.path.join(plots_dir, "fps_over_time.png"))

    # รายงาน
    names_map = model.names if hasattr(model, "names") else {}
    total_frames = frame_idx
    total_dets = int(sum(det_counts))
    mean_dt = float(np.mean(proc_times)) if proc_times else 0.0
    avg_proc_fps = len(proc_times)/sum(proc_times) if proc_times else 0.0
    top_classes = ""
    if len(class_count)>0:
        top_pairs = class_count.most_common(10)
        top_classes = "\n".join([f"- {names_map.get(cid, str(cid))}: {cnt}" for cid,cnt in top_pairs])

    report = f"""# YOLOv8 Video Analytics Report

**Model**: `{os.path.basename(MODEL_PATH)}`  
**Source video**: `{os.path.basename(VIDEO_PATH)}`  
**Output video**: `annotated.mp4`  
**Frames**: {total_frames}  
**Estimated source FPS**: {fps_src:.2f}  
**Processing mean Δt/frame**: {mean_dt:.4f} s  
**Processing average FPS**: {avg_proc_fps:.2f}  
**Total detections**: {total_dets}  
**imgsz**: {IMGSZ}, **conf**: {CONF}, **iou**: {IOU}

## Top classes (by count)
{top_classes if top_classes else "- (no detections)"}

## Saved plots
- plots/detections_over_time.png
- plots/confidence_hist.png
- plots/class_counts.png
- plots/box_area_hist.png
- plots/fps_over_time.png

## Per-frame CSV
- csv/per_frame_stats.csv

> หมายเหตุ: กราฟเหล่านี้คือสถิติจากผลทำนาย ไม่ใช่ PR/mAP (ต้องมี label จริงจึงจะวัดได้)
"""
    with open(os.path.join(save_root, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print("\n=== เสร็จสิ้น 🎉 ===")
    print("โฟลเดอร์ผลลัพธ์:", os.path.abspath(save_root))
    print("วิดีโอใส่กรอบ:", os.path.abspath(out_video_path))
    if AUTO_OPEN_FOLDER:
        try:
            import webbrowser
            webbrowser.open(os.path.abspath(save_root))
        except Exception:
            pass

if __name__ == "__main__":
    main()
