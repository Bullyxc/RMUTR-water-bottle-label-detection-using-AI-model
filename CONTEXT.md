# CONTEXT.md

## ภาพรวม

โปรเจค **RMUTR Bottle** เป็น Mini Project ด้าน Computer Vision สำหรับตรวจสอบขวดน้ำบนสายพานการผลิต โดยใช้ YOLOv8 Object Detection ตรวจขวดและฉลากก่อนเข้าสู่กระบวนการอบไอน้ำ ซึ่งจะทำให้ฉลากหดเข้ากับรูปทรงขวด

เอกสารอธิบายโครงการฉบับเต็มอยู่ที่ `RMUTR Bottle/docs/` และภาพประกอบที่ใช้ในเอกสารเรียงตามลำดับอยู่ใน `assets/1.jpg` ถึง `assets/34.jpg`

## ปัญหาและคลาสข้อมูล

โมเดลต้องแยกสถานะของขวด/ฉลาก 3 กลุ่ม:

| คลาส | ความหมาย |
|---|---|
| `OK` | ขวดมีฉลากสมบูรณ์ และขอบล่างของฉลากไม่ต่ำเกินขอบสีเขียว |
| `NG` | ฉลากไม่สมบูรณ์ ขอบล่างต่ำเกินขอบสีเขียว หรือขวดไม่มีฉลาก |
| `Label leak` | ฉลากขวดน้ำหลุด |

## Pipeline ตามเอกสาร

1. เก็บข้อมูลจากวิดีโอในโรงงาน แล้วใช้ Python/OpenCV จับภาพแบบเต็มเฟรมและ Crop บริเวณขวดทุก 3 วินาที
2. นำภาพที่แตกต่างกันประมาณ 300 ภาพเข้า Roboflow และ Label เป็น `OK`, `NG`, `Label leak`
3. แบ่งข้อมูลเป็น Train/Validation/Test ในสัดส่วน 70%/20%/10%
4. ทำ preprocessing ใน Roboflow: Auto-Orient, Resize เป็น 640x640 และ Auto-Adjust Contrast
5. ทำ augmentation: Flip, Rotation, Shear, Brightness, Exposure และ Blur
6. ฝึก YOLOv8s บน Google Colab ด้วย T4 GPU โดยเอกสารระบุ `epochs=400` และ `imgsz=640`
7. ดาวน์โหลด `runs.zip` เพื่อดูผลลัพธ์ เช่น Loss, mAP, precision และ recall
8. ทดสอบโมเดลกับภาพใหม่ผ่านขั้นตอน `Inference with Custom Model`

วิดีโอ Inference/ผลการทำนาย: <https://youtu.be/NY6tvybFny8>

ลิงก์นี้เป็นคลิปผลการทำนายของโมเดล ไม่ใช่คลิปวิดีโอดิบที่ถ่ายจากโรงงาน

## โครงสร้างใน repository

- `README.md` - เอกสาร GitHub ที่ถอดเนื้อหาและรูปภาพจาก PDF/DOCX พร้อมขั้นตอนติดตั้งและรัน
- `AGENT.md` - แนวทางสำหรับ coding agent/ผู้ดูแล repository
- `assets/` - ภาพประกอบเอกสาร 34 ภาพ และไฟล์ `Inferring_video_link`
- `RMUTR Bottle/*.py` - สคริปต์เก็บภาพ ฝึกโมเดล ตรวจจับ วิเคราะห์ผล และ utility
- `RMUTR Bottle/backgroundremover/` - โค้ด U2Net/background removal ที่ vendored ไว้เป็นชุดเสริม
- `RMUTR Bottle/yolov8n.pt`, `yolov8s.pt` - น้ำหนัก YOLO ที่มีอยู่ใน repository
- `RMUTR Bottle/run2/`, `runs/`, `runs3/` - artifacts จากการฝึกหลายรอบ พร้อม `args.yaml`, plots และ weights
- `RMUTR Bottle/Testresult01/` - ตัวอย่างผลจาก `Testyolo_video_analytics.py` มี report, plots และ CSV

## สถานะและข้อควรระวังของโค้ดปัจจุบัน

- ยังไม่มี `requirements.txt` หรือ `pyproject.toml`; README จึงระบุคำสั่งติดตั้งจาก imports ที่พบใน source
- สคริปต์หลักใช้ hard-coded paths และแก้ค่าได้ที่บล็อก `CONFIG` ด้านบนไฟล์
- วิดีโออินพุต `MVI_3018.MP4` และ `MVI_3019.MP4` ไม่ได้อยู่ใน repository ปัจจุบัน
- `test_detect_video_noargs.py` และ `Testyolo_video_analytics.py` อ้าง `runs2/...` แต่โฟลเดอร์ที่มีอยู่คือ `run2/`; ต้องตรวจ path ก่อนรัน
- `test_roi_noargs.py` อ้าง `runs/detect/train/weights/...` แต่ artifact เดิมใน `runs/` ใช้โฟลเดอร์ `weight/` (เอกพจน์)
- `Testdetection SHortcut.py` อ้าง `runs3 (1)/...` ซึ่งไม่มีอยู่ใน repository ปัจจุบัน
- `test_swim01.py` อ้าง `bestswim01.pt` ซึ่งไม่มีอยู่ใน repository ปัจจุบัน
- `trainmodel GPU.py` เป็น local training example ที่ใช้ `yolov8m.pt`, `data.yaml`, 50 epochs, `imgsz=240`, `batch=32` ไม่ใช่ค่าการฝึก YOLOv8s ในเอกสาร
- `runs3/detect/train/args.yaml` สะท้อนการฝึกหนึ่งรอบที่ใช้ `yolov8s.pt`, 400 epochs และ `imgsz=640`; `run2` และ `runs` เป็นผลจาก configuration อื่น
- `backgroundremover/cmd/cli.py` import `distutils.util` ซึ่งอาจใช้ไม่ได้บน Python 3.12+ และชุดนี้ไม่ถูกเรียกโดย detection pipeline หลัก
- การรัน detection/training จะสร้างไฟล์จำนวนมากและอาจใช้ GPU; ควรรันแบบตั้งใจและตรวจ output directory ก่อนทุกครั้ง

## ผลลัพธ์ที่คาดหวัง

สำหรับ capture scripts:

- full-frame images จะถูกบันทึกใน `Cropped bottle fullframe01/`
- crop images จะถูกบันทึกในโฟลเดอร์ที่กำหนดโดย `OUT_DIR` เช่น `Cropped bottle03/`

สำหรับ `Testyolo_video_analytics.py`:

```text
Testresult01/
└── run_<YYYYMMDD_HHMMSS>/
    ├── annotated.mp4
    ├── csv/per_frame_stats.csv
    ├── plots/
    │   ├── detections_over_time.png
    │   ├── confidence_hist.png
    │   ├── class_counts.png
    │   ├── box_area_hist.png
    │   └── fps_over_time.png
    └── report.md
```

รายงาน analytics เป็นสถิติจากผลทำนาย เช่น จำนวน detection, confidence และ FPS ไม่ใช่ค่า PR/mAP จาก ground truth โดยตรง หากต้องการประเมิน accuracy ต้องใช้ validation/test labels และคำสั่งประเมินของ YOLO ให้ถูกชุดข้อมูล

## แหล่งอ้างอิงโครงการ

- [YOLOv8 custom dataset notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb)
- [Roboflow RMUTR dataset](https://app.roboflow.com/kittipat-blwh5/rmutr-salaya-bottle-label-hybrid-ibceo/2)
