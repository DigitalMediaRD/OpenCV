import cv2
import numpy as np
def ImageRead():
    #img=cv2.imread('res/img001.png')
    #img = cv2.imread('res/img001.png',cv2.IMREAD_UNCHANGED)
    #img = cv2.imread('res/img001.png',cv2.IMREAD_GRAYSCALE)
    #img = cv2.imread('res/img001.png',cv2.IMREAD_COLOR)
    #img = cv2.imread('res/img001.png',cv2.IMREAD_ANYCOLOR)
    img = cv2.imread('res/img001.png',cv2.IMREAD_REDUCED_COLOR_4)
    print(img)
    return img





def ImageWrite():
    nparray=np.random.default_rng()
    img = nparray.integers(-255, 255, size=(700, 700))
    cv2.imwrite('res/img001-w.png',img)
    print(img)

def ImageShow():

    cv2.imshow('Show',ImageRead())
    cv2.waitKey(0)

ImageShow()
