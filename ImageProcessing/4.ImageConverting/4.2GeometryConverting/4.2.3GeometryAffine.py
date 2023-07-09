import cv2
import numpy as np
img=cv2.imread('res/img001.png')
cv2.imshow('show',img)
height=img.shape[0]
width=img.shape[1]
dsize=(width,height)



print(img.shape[0])
def Translation():
    m = np.float32([[1, 0, -100], [0, 1, 50]])
    img2 = cv2.warpAffine(img, m, dsize)
    cv2.imshow('Translation', img2)
    cv2.waitKey(0)


def Scale():
    m=np.float32([[1,0,0],[0,0.5,0]])
    img2=cv2.warpAffine(img,m,dsize)
    cv2.imshow('Scale',img2)

    cv2.waitKey(0)

def Rotation():
    m=cv2.getRotationMatrix2D((width/2,height/2),-60,0.5)
    img2=cv2.warpAffine(img,m,dsize)
    cv2.imshow('Rotation', img2)

    cv2.waitKey(0)


def Mapping():
    input=np.float32([[0,0],[width-10,0],[0,height-1]])
    output=np.float32([[50,50],[width-100,80],[100,height-100]])
    m=cv2.getAffineTransform(input,output)
    img2=cv2.warpAffine(img,m,dsize)
    cv2.imshow('Mapping',img2)
    cv2.waitKey(0)

Mapping()



