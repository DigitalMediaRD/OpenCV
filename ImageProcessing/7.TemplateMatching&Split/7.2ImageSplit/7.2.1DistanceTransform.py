import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('res/img001.png')
cv2.imshow('original', img)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,imgthresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

kernel=np.ones((3,3),np.uint8)
imgopen=cv2.morphologyEx(imgthresh,cv2.MORPH_OPEN,kernel,iterations=2)

imgbg=cv2.dilate(imgopen,kernel,iterations=3)
imgdist=cv2.distanceTransform(imgopen,cv2.DIST_L2,5)
cv2.imshow('distance',imgdist)


'''
查阅官方文档
'''
ret,imgfg=cv2.threshold(imgdist,0.7*imgdist.max(),255,2)

imgfg=np.uint8(imgfg)
ret,markers=cv2.connectedComponents(imgfg)
unknow=cv2.subtract(imgbg,imgfg)
markers=markers+1
markers[unknow==255]=0
imgwater=cv2.watershed(img,markers)
plt.imshow(imgwater)
plt.title('watershed')
plt.axis('off')
plt.show()
img[imgwater==-1]=[0,255,0]


cv2.imshow('watershed',img)
cv2.waitKey(0)