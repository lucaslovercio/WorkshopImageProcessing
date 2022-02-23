import cv2
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from functionAnisotropicDiffusion import functionAnisotropicDiffusion

path_image = "LumbarSpinalStenosis_case1_1_8.png"
#path_image = "Ultrasound_Scan_ND_142655_1438570_cr.png"

#opening and filtered
imageITK = sitk.ReadImage(path_image, sitk.sitkUInt8)
image = sitk.GetArrayFromImage(imageITK)

#Filtering
median_flag = False
if median_flag:
    size_window = 7
    image_filtered_1 = cv2.medianBlur(image,size_window)
    image_filtered = np.dstack((image_filtered_1,image_filtered_1,image_filtered_1))
else:
    image_filtered, image_filtered_1 = functionAnisotropicDiffusion(imageITK)

#Thresholding
#image_thresholded = (image_filtered > 60) * 255
image_thresholded = np.logical_and(image_filtered > 60, image_filtered < 110) * 255

plt.subplot(131),plt.imshow(image, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(image_filtered_1, cmap='gray'),plt.title('Filtered')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(image_thresholded),plt.title('Thresholding')
plt.xticks([]), plt.yticks([])
plt.show()
