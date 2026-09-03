from ultralytics import YOLO
import os
import torch

# ฟังก์ชันนี้จะครอบโค้ดการเทรนหลักของเราไว้
def start_training():
    """
    ฟังก์ชันสำหรับตั้งค่าและเริ่มการเทรนโมเดล YOLO
    """
    # ตรวจสอบว่ามี GPU ที่รองรับ CUDA หรือไม่
    if torch.cuda.is_available():
        print(f"พบ GPU: {torch.cuda.get_device_name(0)}")
        device = 0  # กำหนดให้ใช้ GPU ตัวแรก
    else:
        print("ไม่พบ GPU, ทำการเทรนด้วย CPU (อาจใช้เวลานาน)")
        device = 'cpu'

    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8m.pt')

    # Path to your data.yaml file
    data_yaml_path = 'data.yaml'

    # ตรวจสอบว่าไฟล์ data.yaml มีอยู่จริงหรือไม่
    if not os.path.exists(data_yaml_path):
        print(f"Error: ไม่พบไฟล์ data.yaml ที่ตำแหน่ง {data_yaml_path}")
    else:
        # Train the model with your custom dataset and configuration
        results = model.train(
            data=data_yaml_path,
            epochs=50,
            imgsz=240,
            batch=32,
            name='yolov8_custom_model_v1',
            device=device
        )

# --- นี่คือส่วนที่สำคัญที่สุด ---
# โค้ดจะเริ่มทำงานจากตรงนี้เมื่อรันสคริปต์โดยตรง
if __name__ == '__main__':
    start_training()