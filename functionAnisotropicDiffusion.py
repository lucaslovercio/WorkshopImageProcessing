import SimpleITK as sitk #not installed in Day 1
import numpy as np

def functionAnisotropicDiffusion(image):
    number_iterations = 800
    time_step = 0.1
    conductance_param = 0.7

    perona_malik_filter = sitk.GradientAnisotropicDiffusionImageFilter()

    perona_malik_filter.SetNumberOfIterations(number_iterations)
    perona_malik_filter.SetTimeStep(time_step)
    perona_malik_filter.SetConductanceParameter(conductance_param)

    dst = perona_malik_filter.Execute(sitk.Cast(image, sitk.sitkFloat32))
    dst = sitk.Cast(dst, sitk.sitkUInt8)

    dst = sitk.GetArrayFromImage(dst)
    dst3 = np.dstack((dst,dst,dst))
    return dst3,dst