import cv2
from screeninfo import get_monitors

# # Load the image
# image_path = 'path/to/your/large_image.jpg'
# image = cv2.imread(image_path)
# if image is None:
#     print("Error: Image not found.")
#     exit()

# Get monitor resolution
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height
print(monitor)

# # Get image dimensions
# image_height, image_width = image.shape[:2]
#
# # Calculate the scaling factor to fit the image on the screen
# # The ratio must be less than 1.0
# scale_width = screen_width / image_width
# scale_height = screen_height / image_height
# scale = min(scale_width, scale_height)
#
# # Check if the image is already smaller than the screen
# if scale >= 1.0:
#     print("Image is already smaller than or equal to screen size. Displaying without resizing.")
#     resized_image = image
# else:
#     # Resize the image while maintaining aspect ratio
#     new_width = int(image_width * scale)
#     new_height = int(image_height * scale)
#     resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
#
# # Display the image
# cv2.imshow("Fitted Image", resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()