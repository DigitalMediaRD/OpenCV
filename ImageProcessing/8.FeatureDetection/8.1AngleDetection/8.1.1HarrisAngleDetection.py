import cv2
import numpy as np
img = cv2.imread('res/img002.png')
cv2.imshow('Original',img)
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
gray=np.float32(gray)
dst=cv2.cornerHarris(gray,2,7,0.01)
img[dst>0.02*dst.max()]=[0,0,255]
cv2.imshow('dst',img)
cv2.waitKey(0)