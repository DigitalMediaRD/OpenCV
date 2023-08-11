import cv2
import matplotlib.pyplot as plt
img = cv2.imread('res/img001.png')
cv2.imshow('original', img)
img2=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hist=cv2.calcHist([img2],[0,1],None,[180,256],[0,180,0,256])
cv2.imshow('2Dhist',hist)



def MatplotShow(hist):
    plt.imshow(hist,interpolation='nearest')
    plt.show()


MatplotShow(hist)