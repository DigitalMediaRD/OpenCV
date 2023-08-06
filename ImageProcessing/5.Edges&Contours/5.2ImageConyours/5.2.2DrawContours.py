import cv2
import numpy as np

img = cv2.imread('res/img001.png')
cv2.imshow('original', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img2 = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)  # 二值化阈值处理
c, h = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓


img3 = np.zeros(img.shape, np.uint8) + 255
img3=cv2.drawContours(img3,c,-1,(100,20,55),2)
cv2.imshow('DrawContours', img3)

cv2.waitKey(0)
cv2.destroyAllWindows()














