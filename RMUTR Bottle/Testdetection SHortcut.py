from ultralytics import YOLO
model = YOLO("runs3 (1)/detect/train/weights/best.pt")
model.predict(source=0, imgsz=640, conf=0.5, show=True, save=False)
