import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('res/img002.png')
fast=cv2.FastFeatureDetector_create()
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
sift=cv2.SIFT_create()
kp=sift.detect(gray,None)
img2=cv2.drawKeypoints(img,kp,None,flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

plt.imshow(img2)
plt.axis('off')
plt.show()