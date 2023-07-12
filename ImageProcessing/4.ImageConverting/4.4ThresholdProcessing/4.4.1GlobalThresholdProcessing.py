import cv2
img=cv2.imread('res/img001.png')
cv2.imshow('img',img)

'''
二值化阈值处理
将大于阈值的像素值更改为255，其余像素值更改为0
'''
def Thresh_Binart(input):
    ret,ouput=cv2.threshold(input,150,255,cv2.THRESH_BINARY)
    cv2.imshow('imgThresh_Binary',ouput)
    cv2.waitKey(0)

'''
反二值化阈值处理
将大于阈值的像素值更改为0，其余像素值更改为255
'''
def Thresh_Binart_Inv(input):
    ret, ouput = cv2.threshold(input, 150, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('imgThresh_Binary', ouput)
    cv2.waitKey(0)

'''
截断阈值处理
将大于阈值的像素值更改为设置好的阈值，其余像素值不变
'''
def Thresh_Trunc(input):
    ret, ouput = cv2.threshold(input, 100, 255, cv2.THRESH_TRUNC)
    cv2.imshow('imgThresh_Binary', ouput)
    cv2.waitKey(0)

'''
超阈值零处理
将大于阈值的像素值更改为0，其余像素值不变
'''
def Thresh_Tozero_Inv(input):
    ret, ouput = cv2.threshold(input, 100, 255, cv2.THRESH_TOZERO_INV)
    cv2.imshow('imgThresh_Binary', ouput)
    cv2.waitKey(0)

'''
低阈值零处理
将小于阈值的像素值更改为0，其余像素值不变
'''
def Thresh_Tozero(input):
    ret, ouput = cv2.threshold(input, 100, 255, cv2.THRESH_TOZERO)
    cv2.imshow('imgThresh_Binary', ouput)
    cv2.waitKey(0)

'''
Otsu算法阈值零处理
遍历当前图像的所有像素值，再选择最佳阈值
可在此基础上与其他阈值处理方法叠加使用
'''
def Otsu_Thresh(input):
    img=cv2.imread('res/img001.png',cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('Gray',img)
    ret, ouput1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    #cv2.imshow('imgTHRESH_BINARY', ouput1)
    ret, ouput2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('THRESH_BINARY+THRESH_OTSU', ouput2)
    ret, ouput3 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imshow('THRESH_BINARY_INV+THRESH_OTSU', ouput3)
    ret, ouput4 = cv2.threshold(img, 100, 255,  cv2.THRESH_OTSU)
    cv2.imshow('THRESH_OTSU', ouput4)
    cv2.waitKey(0)


'''
三角算法阈值零处理
可在此基础上与其他阈值处理方法叠加使用
'''
def Otsu_Triangle(input):
    img=cv2.imread('res/img001.png',cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('Gray',img)
    ret, ouput1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow('imgTHRESH_BINARY', ouput1)
    ret, ouput2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    cv2.imshow('THRESH_BINARY+THRESH_TRIANGLE', ouput2)
    ret, ouput3 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    cv2.imshow('THRESH_BINARY_INV+THRESH_TRIANGLE', ouput3)

    cv2.waitKey(0)

Otsu_Triangle(img)


