import cv2
import numpy as np
img=cv2.imread('res/img001.png')
cv2.imshow('Original',img)
img2=cv2.bilateralFilter(img,20,100,100)
cv2.imshow('MedianFiltering',img2)
cv2.waitKey(0)