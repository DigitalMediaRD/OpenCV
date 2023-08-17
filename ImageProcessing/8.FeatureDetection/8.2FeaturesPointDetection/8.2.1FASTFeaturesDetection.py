import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('res/img002.png')
fast=cv2.FastFeatureDetector_create()
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
kp=fast.detect(gray,None)
img2=cv2.drawKeypoints(gray,kp,None,color=(0,0,255))
cv2.imshow('FAST points',img2)
fast.setThreshold(20)
kp=fast.detect(gray,None)
n=0
for p in kp:
    n+=1

img3=cv2.drawKeypoints(gray,kp,None,color=(0,0,255))
cv2.imshow('Threshold20',img3)
cv2.waitKey(0)