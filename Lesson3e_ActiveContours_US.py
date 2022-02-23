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
    size_window = 7
    image_filtered_1 = cv2.medianBlur(image,size_window)
    image_filtered = np.dstack((image_filtered_1,image_filtered_1,image_filtered_1))
else:
    image_filtered, image_filtered_1 = functionAnisotropicDiffusion(imageITK)

#intial contour

#Adapted from https://scikit-image.org/docs/dev/auto_examples/edges/plot_active_contours.html
#snake algorithm contour
gaussian_filtered_image = gaussian(image_filtered_1, 2, preserve_range=False)

yx_init = [41,64]
yx_end = [116,384]

#Far edge
#yx_init = [125,241]#yx_init = [105,48]
#yx_end = [168,410]

r = np.linspace(yx_init[0], yx_end[0], 5000)
c = np.linspace(yx_init[1], yx_end[1], 5000)
init = np.array([r, c]).T

snake = active_contour(gaussian_filtered_image,
                       init, boundary_condition='fixed',
                       alpha=0.1, beta=1.0, w_line=0, w_edge=1, gamma=0.1,
                       max_num_iter=300, convergence=0.1)

plt.figure(1)
plt.subplot(231),plt.imshow(image, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(image_filtered_1, cmap='gray'),plt.title('Filtered')
plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(gaussian_filtered_image, cmap='gray'),plt.title('Gauss')
plt.xticks([]), plt.yticks([])

plt.subplot(234)
plt.imshow(image_filtered_1, cmap='gray'),plt.title('Initial contour')
plt.plot(init[:, 1], init[:, 0], '--r', lw=2)
plt.plot(snake[:, 1], snake[:, 0], '-b', lw=2)
plt.xticks([]), plt.yticks([])

plt.show()

