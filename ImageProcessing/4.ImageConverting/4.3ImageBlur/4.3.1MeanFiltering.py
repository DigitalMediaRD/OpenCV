import cv2
img=cv2.imread('res/img001.png')
cv2.imshow('Original',img)


def MeanFiltering():
    img2=cv2.blur(img,(20,20))
    cv2.imshow('ImgBlur',img2)
    cv2.waitKey(0)


MeanFiltering()