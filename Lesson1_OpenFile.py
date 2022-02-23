import cv2

path_image = "LumbarSpinalStenosis_case1_1_8.png"

image = cv2.imread(path_image)

cv2.imshow("slice", image)
cv2.waitKey(0)
#cv2.destroyAllWindows()
