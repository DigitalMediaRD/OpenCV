import cv2
import numpy as np

img=cv2.imread('res/img001.png')
cv2.imshow('img',img)
kernel=np.ones((30,30),np.uint8)
img2=cv2.erode(img,kernel,iterations=10)
cv2.imshow('Erosion',img2)
cv2.waitKey(0)