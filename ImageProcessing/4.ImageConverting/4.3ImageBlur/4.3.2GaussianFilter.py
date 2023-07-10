import cv2
img=cv2.imread('res/img001.png')
cv2.imshow('Original',img)
img2=cv2.GaussianBlur(img,(105,105),0,0)
cv2.imshow('GaussianBlur',img2)
cv2.waitKey(0)