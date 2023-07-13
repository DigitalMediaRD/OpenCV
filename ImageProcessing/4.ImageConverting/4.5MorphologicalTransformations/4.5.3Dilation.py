import cv2
import numpy as np

img=cv2.imread('res/img001.png')
cv2.imshow('img',img)
kernel=np.ones((2,20),np.uint8)
img2=cv2.dilate(img,kernel,iterations=1)
cv2.imshow('Dilation',img2)
cv2.waitKey(0)