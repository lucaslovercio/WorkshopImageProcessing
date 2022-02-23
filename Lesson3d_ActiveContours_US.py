import cv2
import SimpleITK as sitk
import numpy
import numpy as np
from matplotlib import pyplot as plt
from functionAnisotropicDiffusion import functionAnisotropicDiffusion
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

#path_image = "LumbarSpinalStenosis_case1_1_8.png"
path_image = "Ultrasound_Scan_ND_142655_1438570_cr.png"

#opening and filtered
imageITK = sitk.ReadImage(path_image, sitk.sitkUInt8)
image = sitk.GetArrayFromImage(imageITK)

#Filtering
median_flag = True
if median_flag:
    size_window = 5
    image_filtered_temp = cv2.medianBlur(image,size_window)
    image_filtered = np.dstack((image_filtered_temp,image_filtered_temp,image_filtered_temp))
else:
    image_filtered, dest = functionAnisotropicDiffusion(imageITK)

image = np.dstack((image,image,image))

#intial contour

s = np.linspace(0, 2*np.pi, 2000) #2000 nodes
r = 203 + 20*np.sin(s)
c = 162 + 40*np.cos(s)
init = np.array([r, c]).T

#snake algorithm contour
gaussian_filtered_image = gaussian(image_filtered, 1, preserve_range=False)
snake = active_contour(gaussian_filtered_image,
                       init,
                       alpha=0.1, beta=1, w_line=0, w_edge=1, gamma=0.1,
                       max_num_iter=6000, convergence=0.1)

plt.figure(1)
plt.subplot(231),plt.imshow(image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(image_filtered),plt.title('Filtered')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(gaussian_filtered_image),plt.title('Gauss')
plt.xticks([]), plt.yticks([])

plt.subplot(234)
plt.imshow(image_filtered),plt.title('Initial contour')
plt.plot(init[:, 1], init[:, 0], '--r', lw=2)
plt.plot(snake[:, 1], snake[:, 0], '-b', lw=2)
plt.xticks([]), plt.yticks([])

plt.show()

