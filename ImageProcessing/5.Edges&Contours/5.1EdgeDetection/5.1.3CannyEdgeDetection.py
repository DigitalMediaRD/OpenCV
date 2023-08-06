import cv2
img=cv2.imread('res/img001.png')
cv2.imshow('Original',img)
img2=cv2.Canny(img,200,300)
cv2.imshow('Show',img2)
cv2.waitKey(0)