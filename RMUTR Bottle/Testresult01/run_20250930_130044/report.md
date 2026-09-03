# YOLOv8 Video Analytics Report

**Model**: `best.pt`  
**Source video**: `MVI_3018.MP4`  
**Output video**: `annotated.mp4`  
**Frames**: 14082  
**Estimated source FPS**: 59.94  
**Processing mean Δt/frame**: 0.0871 s  
**Processing average FPS**: 11.48  
**Total detections**: 13463  
**imgsz**: 640, **conf**: 0.5, **iou**: 0.45

## Top classes (by count)
- OK: 10373
- NG: 2081
- Label leak: 1009

## Saved plots
- plots/detections_over_time.png
- plots/confidence_hist.png
- plots/class_counts.png
- plots/box_area_hist.png
- plots/fps_over_time.png

## Per-frame CSV
- csv/per_frame_stats.csv

> หมายเหตุ: กราฟเหล่านี้คือสถิติจากผลทำนาย ไม่ใช่ PR/mAP (ต้องมี label จริงจึงจะวัดได้)
