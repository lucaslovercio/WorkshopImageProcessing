import cv2
import numpy as np
from matplotlib import pyplot as plt #not installed in Day 1
from skimage.measure import profile_line #not installed in Day 1


path_image = "Ultrasound_Scan_ND_142655_1438570_cr.png"

image = cv2.imread(path_image)

kernel_sobel_x = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
#kernel_sobel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

#Canny Edge
#Gaussian filtering is "mandatory" in the technique
#dest_canny = cv2.Canny(image,50,200)

dest_sobel = cv2.filter2D(image, -1, kernel_sobel_x)

plt.subplot(121),plt.imshow(image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dest_sobel),plt.title('Edges')
plt.xticks([]), plt.yticks([])
plt.show()

# Extract intensity values along some profile line
#Coordinates = Y,X

init_coord = (108,261)
end_coord = (175,261)

profile_original = profile_line(image, init_coord, end_coord)
profile_filtered = profile_line(dest_sobel, init_coord, end_coord)

plt.plot(profile_original)
plt.plot(profile_filtered)

plt.ylabel('intensity')
plt.xlabel('line path')
plt.show()