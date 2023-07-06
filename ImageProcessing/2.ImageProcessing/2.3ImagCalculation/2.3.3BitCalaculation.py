import cv2
img1=cv2.imread('res/img001.png',cv2.IMREAD_REDUCED_COLOR_2)
img2=cv2.imread('res/img002.png',cv2.IMREAD_REDUCED_COLOR_2)
img3=cv2.bitwise_or(img1,img2)
img4=cv2.bitwise_and(img1,img2)
img5=cv2.bitwise_not(img1,img2)
img6=cv2.bitwise_xor(img1,img2)

cv2.imshow('or',img3)
cv2.imshow('and',img4)
cv2.imshow('not',img5)
cv2.imshow('xor',img6)
cv2.waitKey(0)