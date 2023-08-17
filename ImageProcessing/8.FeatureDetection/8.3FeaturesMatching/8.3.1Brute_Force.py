import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread('res/img001.png')
img2 = cv2.imread('res/img002.png')

orb=cv2.ORB_create()

kp1,des1=orb.detectAndCompute(img1,None)
kp2,des2=orb.detectAndCompute(img2,None)

def Default():
    bf=cv2.BFMatcher_create(cv2.NORM_HAMMING,crossCheck=True)

    ms=bf.match(des1,des2)
    ms=sorted(ms,key=lambda x:x.distance)

    img3=cv2.drawMatches(img1,kp1,img2,kp2,ms[:20],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img3

def KnnMatch():
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)

    ms = bf.knnMatch(des1, des2,k=2)
    good=[]
    for m,n in ms:
        if m.distance<0.75*n.distance:
            good.append(m)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img3
plt.imshow(KnnMatch())
plt.axis('off')
plt.show()