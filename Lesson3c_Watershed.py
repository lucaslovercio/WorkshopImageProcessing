import cv2
import SimpleITK as sitk
import numpy
import numpy as np
from matplotlib import pyplot as plt
from functionAnisotropicDiffusion import functionAnisotropicDiffusion

path_image = "LumbarSpinalStenosis_case1_1_8.png"
#path_image = "Ultrasound_Scan_ND_142655_1438570_cr.png"

#opening and filtered
imageITK = sitk.ReadImage(path_image, sitk.sitkUInt8)
image = sitk.GetArrayFromImage(imageITK)

#Filtering
median_flag = True
if median_flag:
    size_window = 7
    image_filtered_temp = cv2.medianBlur(image,size_window)
    image_filtered = np.dstack((image_filtered_temp,image_filtered_temp,image_filtered_temp))
else:
    image_filtered, dest = functionAnisotropicDiffusion(imageITK)

image = np.dstack((image,image,image))

#Adapted from:
#https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html

#Thresholding
#image_thresholded = (image_filtered > 60) * 255
image_thresholded = np.logical_and(image_filtered > 60, image_filtered < 120) * 255
image_thresholded = image_thresholded.astype(numpy.ushort)
image_thresholded_1 = image_thresholded[:,:, 1]

#Find markers
# Creating kernel
kernel = np.ones((19, 19), np.uint8)
image_th_eroded = cv2.erode(image_thresholded_1, kernel)
# Marker labelling
image_th_eroded = np.uint8(image_th_eroded)
ret, markers = cv2.connectedComponents(image_th_eroded)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
opening = cv2.morphologyEx(image_thresholded_1,cv2.MORPH_OPEN,kernel, iterations = 2)
sure_bg = cv2.dilate(opening,kernel,iterations=3)
sure_bg = np.uint8(sure_bg)
unknown = cv2.subtract(sure_bg,image_th_eroded)
markers[unknown==255] = 0

#Watershed algorithm
image_filtered_marked = image_filtered
markers = cv2.watershed(image_filtered_marked,markers)
image_filtered_marked[markers == -1] = [255,0,0]

#Plot only one object
one_object = markers == 9

num_pixels_object = np.count_nonzero(one_object)

plt.figure(1)
plt.subplot(131),plt.imshow(image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(image_filtered),plt.title('Filtered')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(image_thresholded),plt.title('Thresholding')
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(2)
plt.subplot(141),plt.imshow(image_th_eroded),plt.title('Eroded')
plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(markers),plt.title('Watershed result')
plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(image_filtered_marked),plt.title('Watershed result')
plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(one_object),plt.title('One Object Px:' + str(num_pixels_object))
plt.xticks([]), plt.yticks([])
plt.show()
