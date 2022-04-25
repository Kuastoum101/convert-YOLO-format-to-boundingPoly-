import cv2

img = cv2.imread("fall001.jpg")
i_height,i_width = img.shape[:2] #firt two dimeansion
window_name = 'cerceve'

# YOLO DEGERLERI (0 0.506757 0.548230 0.445946 0.241148)

                #<object-class> <x> <y> <width> <height>
'''Array umpack'''
center_x, center_y, width, height = [0.506757, 0.548230, 0.445946, 0.241148]
print(center_x)
# YOLO DAN BBOX FORMATINA GITME YOLU
min_x = int(i_width * (center_x - (width/2)))
min_y = int(i_height * (center_y - (height/2)))
max_x = int(i_width * width + min_x)
max_y = int(i_height * height + min_y)
#bbox_row = str(f"{min_x} {max_x} {min_y} {max_y}")

# Start coordinate, here (5, 5)
# represents the top left corner of rectangle
sol_ust = (min_x, min_y)

# Ending coordinate, here (220, 220)
# represents the bottom right corner of rectangle
sag_alt = (max_x, max_y)

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# Using cv2.rectangle() method
# Draw a rectangle with blue line borders of thickness of 2 px
img = cv2.rectangle(img, sol_ust, sag_alt, color, thickness)


cv2.imshow(window_name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()