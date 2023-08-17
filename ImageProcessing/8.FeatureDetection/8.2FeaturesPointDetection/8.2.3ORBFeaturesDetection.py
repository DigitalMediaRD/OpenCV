import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('res/img002.png')

ORB=cv2.ORB_create()
kp=ORB.detect(img,None)
img2=cv2.drawKeypoints(img,kp,None,color=(0,0,255))


cv2.imshow('ORB',img2)
cv2.waitKey(0)