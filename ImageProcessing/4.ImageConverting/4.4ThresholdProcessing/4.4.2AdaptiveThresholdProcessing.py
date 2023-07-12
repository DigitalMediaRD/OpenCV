import cv2
img=cv2.imread('res/img001.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('img',img)
img2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,10)
cv2.imshow('AdaptiveImg',img2)
cv2.waitKey(0)

