import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread('res/img001.png')
cv2.imshow('original', img1)
temp=cv2.imread('res/template.png')
cv2.imshow('template',temp)

img1gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY,dstCn=1)
tempgray=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY,dstCn=1)
th,tw=tempgray.shape

img1h,img1w=img1gray.shape
res=cv2.matchTemplate(img1gray,tempgray,cv2.TM_SQDIFF_NORMED)
mloc=[]
threshold=0.24
for i in range(img1h-th):
    for j in range(img1w-tw):
        if res[i][j]<=threshold:
            mloc.append((i,j))

for pt in mloc:
    cv2.rectangle(img1,pt,(pt[0]+tw,pt[1]+th),(255,0,0),2)

cv2.imshow('result',img1)
cv2.waitKey(0)