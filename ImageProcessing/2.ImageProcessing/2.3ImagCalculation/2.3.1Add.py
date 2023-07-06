import cv2
img1=cv2.imread('res/img001.png',cv2.IMREAD_REDUCED_COLOR_2)
img2=cv2.imread('res/img002.png',cv2.IMREAD_REDUCED_COLOR_2)




def AddCalculation(img1,img2):
    return img1+img2


def CVAddCalculation(img1,img2):
    img=cv2.add(img1,img2)
    return img





Addimg=AddCalculation(img1,img2)
CVAddimg=CVAddCalculation(img1,img2)


cv2.imshow('1',img1)
cv2.imshow('2',img2)
cv2.imshow('Add',Addimg)
cv2.imshow('CVAdd',CVAddimg)
cv2.waitKey(0)
