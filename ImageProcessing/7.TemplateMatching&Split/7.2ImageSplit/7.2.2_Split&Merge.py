import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread('res/img001.png')
img2 = cv2.imread('res/img002.png')
img=img1.copy()
img1Gaus=[img]
for i in range(6):
    img=cv2.pyrDown(img)
    img1Gaus.append(img)

img=img2.copy()
img2Gaus=[img]
for i in range(6):
    img=cv2.pyrDown(img)
    img2Gaus.append(img)

img1Laps=[img1Gaus[5]]
for i in range(5,0,-1):
    img=cv2.pyrUp(img1Gaus[i])
    lap=cv2.subtract(img1Gaus[i-1],img)
    img1Laps.append(lap)

img2Laps=[img2Gaus[5]]
for i in range(5,0,-1):
    img=cv2.pyrUp(img2Gaus[i])
    lap=cv2.subtract(img2Gaus[i-1],img)
    img2Laps.append(lap)

imgLaps=[]
for la,lb in zip(img1Laps,img2Laps):
    rows,cols,dpt=la.shape
    ls=la.copy()
    ls[:,int(cols/2):]=lb[:,int(cols/2):]
    imgLaps.append(ls)

img=imgLaps[0]
for i in range(1,6):
    img=cv2.pyrUp(img)
    img=cv2.add(img,imgLaps[i])

direct=img1.copy()
direct[:,int(cols/2):]=img2[:,int(cols/2):]
cv2.imshow('Direct',direct)
cv2.imshow('Pyramud',img)
cv2.waitKey(0)