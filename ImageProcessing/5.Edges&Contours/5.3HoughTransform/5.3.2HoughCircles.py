import cv2
import numpy as np
img = cv2.imread('res/img001.png')
cv2.imshow('original', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,50,150,apertureSize=3)
circles=cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,50,param2=30,minRadius=10,maxRadius=40)
circles=np.uint16(np.around(circles))
img2=img.copy()
for i in circles[0,:]:
    # 画圆
    cv2.circle(img2,(i[0],i[1]),i[2],(255,0,0),2)
    # 画圆心
    cv2.circle(img2, (i[0], i[1]), 2, (255, 0, 0), 3)


cv2.imshow('HoughLines',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()