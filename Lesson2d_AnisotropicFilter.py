import SimpleITK as sitk #not installed in Day 1
from matplotlib import pyplot as plt #not installed in Day 1
from skimage.measure import profile_line #not installed in Day 1
import numpy as np

image = sitk.ReadImage("Ultrasound_Scan_ND_142655_1438570_cr.png", sitk.sitkUInt8)


number_iterations = 500
time_step = 0.1
conductance_param = 0.7

perona_malik_filter = sitk.GradientAnisotropicDiffusionImageFilter()

perona_malik_filter.SetNumberOfIterations(number_iterations)
perona_malik_filter.SetTimeStep(time_step)
perona_malik_filter.SetConductanceParameter(conductance_param)

dst = perona_malik_filter.Execute(sitk.Cast(image, sitk.sitkFloat32))
dst = sitk.Cast(dst, sitk.sitkUInt8)

image = sitk.GetArrayFromImage(image)
print(str(image.shape))
image = np.dstack((image,image,image))
print(str(image.shape))
dst = sitk.GetArrayFromImage(dst)
dst = np.dstack((dst,dst,dst))

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