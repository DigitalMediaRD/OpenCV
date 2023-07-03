import cv2
import numpy as np
def ImageRead():
    img=cv2.imread('res/img001.png')
    print(img)





def ImageWrite():
    nparray=np.random.default_rng()
    img = nparray.integers(-255, 255, size=(700, 700))
    cv2.imwrite('res/img001-w.png',img)
    print(img)




ImageWrite()
