import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('res/img001.png')
cv2.imshow('original', img)

histb,e1= np.histogram(img[0].ravel(),[256],[0,255])
histg ,e2= np.histogram(img[1].ravel(),[256],[0,255])
histr ,e3= np.histogram(img[2].ravel(),[256],[0,255])

print('读取图像数据失败')
# 返回值edge
plt.plot(histb,color='b')
plt.plot(histg,color='g')
plt.plot(histr,color='r')
plt.show()

