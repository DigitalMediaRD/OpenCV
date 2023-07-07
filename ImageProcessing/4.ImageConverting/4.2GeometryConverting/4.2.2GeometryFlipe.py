import cv2
img=cv2.imread('res/img001.png')
cv2.imshow('showing',img)
while True:
    key=cv2.waitKey()
    if key==48:
        img2=img
    elif key==49:
        img2=cv2.flip(img,0)
    elif key==50:
        img2=cv2.flip(img,1)
    elif key==51:
        img2=cv2.flip(img,-1)
    else:
        break
    cv2.imshow('Result',img2)