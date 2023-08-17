import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread('res/img001.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('res/img002.png',cv2.IMREAD_GRAYSCALE)

orb=cv2.ORB_create()

kp1,des1=orb.detectAndCompute(img1,None)
kp2,des2=orb.detectAndCompute(img2,None)

FLANN_INDEX_LSH = 6
index_params= dict(algorithm =FLANN_INDEX_LSH,table_number = 6,key_size = 12,multi_probe_level = 1)

FLANN_INDEX_LSH=6
index_params=dict(algorithm=FLANN_INDEX_LSH,table_number=6,key_size=12,multi_probe_level=1)
search_params = dict(checks=50)
flann=cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.match(des1,des2)
draw_params = dict(matchColor=(0,255,0),singlePointColor=(255,0,0),matchesMask = None,flags = cv2.DrawMatchesFlags_DEFAULT)
img3 = cv2.drawMatches(img1,kp1,img2,kp2, matches[:20],None,**draw_params)

plt.imshow(img3)
plt.axis('off')
plt.show()