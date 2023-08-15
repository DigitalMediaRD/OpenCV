import cv2
import numpy as np
import matplotlib.pyplot as plt
img0 = cv2.imread('res/img001.png')
img1=cv2.pyrDown(img0)
img2=cv2.pyrDown(img1)
img3=cv2.pyrDown(img2)
def Lsubract():
    imgL0=cv2.subtract(img0,cv2.pyrUp(img1))
    imgL1=cv2.subtract(img1,cv2.pyrUp(img2))
    imgL2=cv2.subtract(img2,cv2.pyrUp(img3))
    cv2.imshow('imgL0', imgL0)
    cv2.imshow('imgL1', imgL1)
    cv2.imshow('imgL2', imgL2)
cv2.imshow('img0', img0)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
print('0 level: ',img0.shape)
print('1 level: ',img1.shape)
print('2 level: ',img2.shape)
cv2.waitKey(0)
Lsubract()
cv2.waitKey(0)