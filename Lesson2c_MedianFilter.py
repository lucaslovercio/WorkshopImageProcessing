import cv2
import numpy as np
from matplotlib import pyplot as plt #not installed in Day 1
from skimage.measure import profile_line #not installed in Day 1


path_image = "Ultrasound_Scan_ND_142655_1438570_cr.png"

image = cv2.imread(path_image)

size_window = 5

dst = cv2.medianBlur(image,size_window)

plt.subplot(121),plt.imshow(image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Filtered')
plt.xticks([]), plt.yticks([])
plt.show()

# Extract intensity values along some profile line
#Coordinates = Y,X

init_coord = (108,261)
end_coord = (175,261)

profile_original = profile_line(image, init_coord, end_coord)
profile_filtered = profile_line(dst, init_coord, end_coord)

plt.plot(profile_original)
plt.plot(profile_filtered)

plt.ylabel('intensity')
plt.xlabel('line path')
plt.show()