import cv2
img=cv2.imread('res/img001.png')
cv2.imshow('Original',img)
img2=cv2.boxFilter(img,16,(3,3),normalize=False)
cv2.imshow('BoxFilter',img2)
cv2.waitKey(0)