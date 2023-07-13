import cv2
import numpy as np

img=cv2.imread('res/img001.png')
cv2.imshow('img',img)
kernel=np.ones((5,5),np.uint8)

def OpenOperation():
    op=cv2.MORPH_OPEN
    img2=cv2.morphologyEx(img,op,kernel,iterations=5)
    cv2.imshow('img2',img2)
    cv2.waitKey(0)

def CloseOperation():
    op = cv2.MORPH_CLOSE
    img2 = cv2.morphologyEx(img, op, kernel, iterations=5)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)


def MorphologicalGradient():
    op = cv2.MORPH_GRADIENT
    img2 = cv2.morphologyEx(img, op, kernel, iterations=1)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)

def BlackHat():
    op = cv2.MORPH_BLACKHAT
    img2 = cv2.morphologyEx(img, op, kernel, iterations=1)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)


def TopHat():
    op = cv2.MORPH_TOPHAT
    img2 = cv2.morphologyEx(img, op, kernel, iterations=1)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)


