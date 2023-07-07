import cv2
img=cv2.imread('res/img001.png')
sc=[1,0.2,1.5,3]
cv2.imshow('Original',img)
while True:
    key=cv2.waitKey()

    if 48<=key<=52:
        # key为获取键盘按键的ASCII值，上诉ASCII取值范围转换为键盘按键则为数字0~数字4
        x=y=sc[key-48]
        img2=cv2.resize(img,None,fx=x,fy=y)
        cv2.imshow('Showing',img2)