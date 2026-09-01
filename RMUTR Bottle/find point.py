import cv2

# --- การตั้งค่า ---
VIDEO_PATH = 'MVI_3018.MP4'  # <<< แก้ไขเป็นชื่อไฟล์วิดีโอของคุณ
WINDOW_NAME = 'Video with Cursor Coordinates'

# ตัวแปรสำหรับเก็บพิกัดของเมาส์ล่าสุด
mouse_coords = (0, 0)


# ฟังก์ชัน Callback ที่จะถูกเรียกเมื่อมีการขยับเมาส์
def get_mouse_position(event, x, y, flags, param):
    """อัปเดตพิกัดของเมาส์เมื่อมีการเคลื่อนไหว"""
    global mouse_coords
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_coords = (x, y)


# --- ส่วนการทำงานหลัก ---

# 1. เปิดไฟล์วิดีโอ
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"เกิดข้อผิดพลาด: ไม่สามารถเปิดไฟล์วิดีโอ '{VIDEO_PATH}' ได้")
    exit()

# 2. สร้างหน้าต่างและผูกฟังก์ชัน mouse callback เข้ากับหน้าต่างนี้
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, get_mouse_position)

print("กำลังเล่นวิดีโอ... กด 'q' เพื่อออกจากโปรแกรม")

while cap.isOpened():
    # 3. อ่านเฟรมจากวิดีโอทีละเฟรม
    ret, frame = cap.read()
    if not ret:
        print("วิดีโอจบแล้ว กำลังเล่นซ้ำ...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # กลับไปที่เฟรมแรกเพื่อเล่นซ้ำ
        continue

    # 4. เตรียมข้อความที่จะแสดงบนหน้าจอ
    # ดึงค่า x และ y จากตัวแปร global ที่ถูกอัปเดตโดย callback
    print(frame.shape)
    x, y = mouse_coords
    text = f'Position: ({x}, {y})'

    # กำหนดค่าต่างๆ สำหรับการวาดข้อความ
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)  # สีขาว
    line_thickness = 2
    text_position = (15, 40)  # พิกัดที่จะวางข้อความ (มุมบนซ้าย)

    # เพิ่มพื้นหลังสีดำเล็กน้อยเพื่อให้อ่านข้อความง่ายขึ้น
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, line_thickness)
    cv2.rectangle(frame, (10, 10), (20 + text_width, 50), (0, 0, 0), -1)

    # 5. วาดข้อความพิกัดลงบนเฟรมวิดีโอ
    cv2.putText(frame, text, text_position, font, font_scale, font_color, line_thickness)

    # 6. แสดงเฟรมที่แก้ไขแล้วในหน้าต่าง
    cv2.imshow(WINDOW_NAME, frame)

    # 7. รอรับการกดปุ่ม 'q' เพื่อออกจากลูป
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 8. ปิดวิดีโอและทำลายหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()

print("ปิดโปรแกรมเรียบร้อย")