import cv2
img=cv2.imread('res/img001.png')
cv2.imshow('Original',img)
img2=cv2.medianBlur(img,7)
cv2.imshow('MedianFiltering',img2)
cv2.waitKey(0)