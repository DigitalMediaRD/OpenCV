import cv2
import matplotlib.pyplot as plt
img = cv2.imread('res/img001.png',0)
cv2.imshow('original', img)
img2=cv2.equalizeHist(img)
cv2.imshow('original', img2)
clahe=cv2.createCLAHE(clipLimit=5)
img3=clahe.apply(img)
cv2.imshow('CLAHE',img3)
cv2.waitKey(0)