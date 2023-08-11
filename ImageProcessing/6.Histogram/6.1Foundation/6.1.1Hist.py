import cv2
import matplotlib.pyplot as plt

img = cv2.imread('res/img001.png')
cv2.imshow('original', img)
plt.hist(img.ravel(),256)
plt.show()