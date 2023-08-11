import cv2
import matplotlib.pyplot as plt
img = cv2.imread('res/img001.png',0)
cv2.imshow('original', img)

plt.figure('Original histogram')
plt.hist(img.ravel(),256)

img2=cv2.equalizeHist(img)
cv2.imshow('equalizeHist', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.figure('EqualizeHist histogram')
plt.hist(img2.ravel(),256)
plt.show()