import cv2
import numpy as np
img=cv2.imread('res/img001.png')
k1=np.array([[3,3,3,3,3],[3,9,9,9,3],[3,11,12,13,3],[3,8,8,8,3],[3,3,3,3,3],])/25
k2=np.ones((5,5),np.float32)/25
img2=cv2.filter2D(img,-1,k1)
cv2.imshow('filter2Dk1',img2)
img2=cv2.filter2D(img,-1,k2)
cv2.imshow('filter2Dk2',img2)
cv2.waitKey(0)