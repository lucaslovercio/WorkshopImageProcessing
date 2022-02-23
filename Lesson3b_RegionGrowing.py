import cv2
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from functionAnisotropicDiffusion import functionAnisotropicDiffusion

path_image = "LumbarSpinalStenosis_case1_1_8.png"
seed = (207,459)

#opening and filtered
imageITK = sitk.ReadImage(path_image, sitk.sitkUInt8)
image = sitk.GetArrayFromImage(imageITK)

#Filtering
median_flag = True
if median_flag:
    size_window = 7
    image_filtered_temp = cv2.medianBlur(image,size_window)
    image_filtered = sitk.GetImageFromArray(image_filtered_temp)
else:
    dst3,dst = functionAnisotropicDiffusion(imageITK)
    image_filtered = sitk.GetImageFromArray(dst)

image = np.dstack((image,image,image))

seg = sitk.Image(imageITK.GetSize(), sitk.sitkUInt8)
seg.CopyInformation(imageITK)
seg = sitk.ConnectedThreshold(image_filtered, seedList=[seed], lower=55, upper=100)
#seg = sitk.NeighborhoodConnected(image_filtered, seedList=[seed], lower=55, upper=100, radius = [3,3])

image_seg = sitk.GetArrayFromImage(seg)

plt.figure(1)
plt.subplot(131),plt.imshow(image),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(sitk.GetArrayFromImage(image_filtered), cmap='gray'),plt.title('Filtered')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(image_seg),plt.title('RG Seg')
plt.xticks([]), plt.yticks([])
plt.show()
