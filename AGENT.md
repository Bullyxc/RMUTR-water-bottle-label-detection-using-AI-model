# AGENT.md

## ขอบเขตของโปรเจค

- Repository root คือโฟลเดอร์ปัจจุบัน (`D:\PycharmProjects\RMUTR Bottle`)
- โค้ด Python และโมเดลอยู่ใน `RMUTR Bottle/`
- เอกสารต้นฉบับอยู่ใน `RMUTR Bottle/docs/`
- ภาพประกอบ README และเอกสารอยู่ใน `assets/`
- `README.md` ที่ root คือเอกสารหลักสำหรับ GitHub

## เป้าหมายของระบบ

โปรเจคนี้เป็น Mini Project สำหรับตรวจจับขวดน้ำและตำแหน่งฉลาก RMUTR ด้วย YOLOv8 Object Detection โดยจำแนกข้อมูลเป็น 3 คลาส:

1. `OK` - ฉลากสมบูรณ์และขอบล่างไม่ต่ำเกินขอบสีเขียว
2. `NG` - ฉลากไม่สมบูรณ์/ขอบล่างต่ำเกินขอบสีเขียว หรือไม่มีฉลาก
3. `Label leak` - ฉลากหลุด

## กฎการทำงานกับ repository

- อ่าน `CONTEXT.md` และ `README.md` ก่อนแก้ไข pipeline หรือชื่อไฟล์สำคัญ
- รันคำสั่งจาก repository root แล้ว `cd "RMUTR Bottle"` เมื่อจะเรียกสคริปต์ในโฟลเดอร์โปรเจค
- สคริปต์ส่วนใหญ่ใช้ค่าคงที่และ path ที่เขียนไว้ด้านบนไฟล์ ไม่มี CLI arguments ให้ถือบล็อก `CONFIG` เป็นจุดตั้งค่าหลัก
- รักษาลำดับและชื่อภาพ `assets/1.jpg` ถึง `assets/34.jpg` เพราะตรงกับภาพประกอบในเอกสารต้นฉบับ
- อย่าใส่ Roboflow API key, download code ที่มี token, วิดีโอจากโรงงาน หรือ credential ใด ๆ ลงใน Git
- อย่าย้าย ลบ หรือ regenerate โฟลเดอร์ข้อมูล/ผลลัพธ์ขนาดใหญ่ (`Cropped bottle*`, `runs*`, weights) โดยไม่มีคำขอชัดเจน
- อย่าใช้ `git reset --hard`, `git checkout --` หรือคำสั่งลบแบบ recursive เพื่อแก้ปัญหาโดยอัตโนมัติ
- ถ้าปรับ README ให้ตรวจลิงก์รูปภาพและ Markdown หลังแก้ไขทุกครั้ง

## Environment ที่คาดหวัง

- โปรเจคนี้ติดตั้งไลบรารีใน Python interpreter ของ PyCharm โดยตรง และไม่ได้ใช้หรือสร้าง virtual environment ภายใน repository; Google Colab ใช้เฉพาะสำหรับ Training Custom Model
- แนะนำ Python 3.11.x สำหรับความเข้ากันได้กับสคริปต์ทั้งหมด โดยเฉพาะ `backgroundremover/cmd/cli.py` ที่ import `distutils`
- dependency หลักที่ติดตั้งใน PyCharm: `torch`, `torchvision`, `ultralytics`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `screeninfo`
- Dataset ใช้ผ่านเว็บไซต์ Roboflow และลิงก์ Dataset จะถูกใช้ในขั้นตอน Training บน Google Colab; ไม่ต้องติดตั้ง Roboflow ใน PyCharm
- dependency เสริมของ `backgroundremover`: `pillow`, `scipy`, `scikit-image`, `pymatting`, `moviepy`, `ffmpeg-python`, `requests`, `flask`, `waitress` และ executable `ffmpeg`/`ffprobe`
- ไม่พบไฟล์ Node.js (`.js`/`.ts`) ใน repository; runnable code ปัจจุบันเป็น Python scripts และโมดูล Python
- repository ยังไม่มี `requirements.txt` หรือ `pyproject.toml`; ใช้คำสั่งติดตั้งที่บันทึกไว้ใน README และเลือก PyTorch build ให้ตรงกับ CUDA ของเครื่อง

## จุดเริ่มต้นที่รันได้

| กลุ่ม | ไฟล์ | หมายเหตุ |
|---|---|---|
| สร้างภาพ | `capture_bottle_fullframe.py` | อ่าน `MVI_3019.MP4`, เก็บ full frame ทุก 3 วินาทีเมื่อพบ `bottle` |
| สร้างภาพ | `capture_bottle_every_sec.py` | อ่าน `MVI_3018.MP4`, crop ขวดทุก 3 วินาที |
| Training | `trainmodel GPU.py` | ต้องมี `yolov8m.pt` และ `data.yaml`; ค่าเริ่มต้น 50 epochs |
| Detection | `test_detect_video_noargs.py` | video detection แบบแสดงผล; default model path เป็น path เก่าที่ต้องตรวจสอบ |
| Detection | `test_roi_noargs.py` | detection บน ROI หลัง resize |
| Analytics | `Testyolo_video_analytics.py` | สร้างวิดีโอผลลัพธ์ กราฟ CSV และ report |
| Utility | `find point.py`, `TestMonitorSize.py` | ช่วยหา ROI และอ่านขนาดจอ |
| Webcam examples | `Testdetection SHortcut.py`, `test_swim01.py` | ใช้ webcam; model path บางตัวไม่มีอยู่ใน repo ปัจจุบัน |
| Optional | `backgroundremover/cmd/cli.py`, `backgroundremover/cmd/server.py` | ลบพื้นหลังภาพ/วิดีโอ ไม่ใช่ pipeline หลัก |

## การตรวจสอบหลังแก้ไข

- ตรวจ syntax ของไฟล์ Python ด้วย `ast.parse` หรือ `python -m py_compile` เฉพาะไฟล์ที่แก้
- ตรวจว่า README อ้างถึงไฟล์ภาพที่มีอยู่จริงด้วยการเช็ก `assets/1.jpg` ถึง `assets/34.jpg`
- อย่ารัน training หรือวิเคราะห์วิดีโอจริงโดยอัตโนมัติ เพราะอาจใช้ GPU/เวลาและสร้างไฟล์ขนาดใหญ่
- ถ้าแก้ path, model, class names หรือ preprocessing ให้บันทึกผลกระทบไว้ใน `CONTEXT.md` หรือ README
