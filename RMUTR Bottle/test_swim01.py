import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (e.g., 'yolov8n.pt' for the nano version)
model = YOLO('bestswim01.pt')

# Initialize the webcam. The '0' indicates the default camera.
# If you have multiple cameras, you might need to change this to 1, 2, etc.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to continuously read frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        break

    # Perform object detection on the frame
    # 'stream=True' is a good practice for video streams as it returns a generator
    results = model(frame, stream=True)

    # Loop through the results to get the annotated frame
    for r in results:
        annotated_frame = r.plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)

    # Press 'q' to quit the live stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()