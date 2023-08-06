import cv2
img=cv2.imread('res/img001.png')
cv2.imshow('Original',img)
img2=cv2.Laplacian(img,cv2.CV_8U)#8位无符号二进制数
cv2.imshow('Show',img2)
cv2.waitKey(0)