import cv2
img=cv2.imread('res/img001.png',cv2.IMREAD_REDUCED_COLOR_2)
cv2.imshow('OriginalImg',img)
def NumpySeparating():
    b=img[:,:,0]
    g=img[:,:,1]
    r=img[:,:,2]
    cv2.imshow('B',b)
    cv2.imshow('G',g)
    cv2.imshow('R',r)
    cv2.waitKey()




def CV2Splite():
    b,g,r=cv2.split(img)
    cv2.imshow('B', b)
    cv2.imshow('G', g)
    cv2.imshow('R', r)
    cv2.waitKey()


def Merge():
    b, g, r = cv2.split(img)
    rgb=cv2.merge([r,g,b])
    gbr=cv2.merge([g,b,r])
    cv2.imshow('RGB', rgb)
    cv2.imshow('GBR', gbr)
    cv2.waitKey()


Merge()