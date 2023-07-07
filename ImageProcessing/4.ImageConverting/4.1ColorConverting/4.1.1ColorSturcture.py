import cv2
img=cv2.imread('res/img001.png')

def RGB():
    cv2.imshow('BGR', img)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('RGB', img2)
    cv2.waitKey(0)


def GRAY():
    cv2.imshow('BGR', img)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('RGB', img2)
    cv2.waitKey(0)

def YCrCb():
    cv2.imshow('BGR', img)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    cv2.imshow('RGB', img2)
    cv2.waitKey(0)




def HSV():
    cv2.imshow('BGR', img)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('RGB', img2)
    cv2.waitKey(0)

HSV()