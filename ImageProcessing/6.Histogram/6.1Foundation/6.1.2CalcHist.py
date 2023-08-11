import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('res/img001.png')
cv2.imshow('original', img)

def CalcHist(img,mask):
    histb = cv2.calcHist([img],[0],mask,[256],[0,255])
    histg = cv2.calcHist([img],[1],mask,[256],[0,255])
    histr = cv2.calcHist([img],[2],mask,[256],[0,255])
    return histb,histg,histr

def CalcHistMask():
    w,h,d=img.shape
    mask=np.zeros((w,h),np.uint8)
    w1=np.int0(w/4)
    w2 = np.int0(w *0.75)
    h1 = np.int0(w / 4)
    h2 = np.int0(w * 0.75)
    mask[w1:w2, h1:h2] = 255
    return mask

histList=CalcHist(img,CalcHistMask())

plt.plot(histList[0],color='b')
plt.plot(histList[1],color='g')
plt.plot(histList[2],color='r')
plt.show()