import cv2
import numpy as np
from matplotlib import pyplot as plt #not installed in Day 1


path_image = "Ultrasound_Scan_ND_142655_1438570_cr.png"

image = cv2.imread(path_image)

size_kernel = 15

#manual kernel
kernel = np.array([[ .1, .1,  .1],
                   [.1, .2, .1],
                   [ .1, .1,  .1]]) # Gx + j*Gy

#Average filter
#kernel = np.ones((size_kernel,size_kernel),np.float32)/ (size_kernel*size_kernel)

#dst = cv2.filter2D(image,-1,kernel) #Depth of the output image [ -1 will give the output image depth as same as the input image]

#Gaussian
dst = cv2.GaussianBlur(image,(size_kernel,size_kernel),0)

plt.subplot(121),plt.imshow(image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Filtered')
plt.xticks([]), plt.yticks([])
plt.show()



#cv2.destroyAllWindows()
