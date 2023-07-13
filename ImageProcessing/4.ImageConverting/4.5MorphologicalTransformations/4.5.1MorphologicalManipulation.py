import cv2
import numpy as np

def TestGetStrurture(index):
    if index==1:
        Matrix=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        print(Matrix)
    elif index==2:
        Matrix = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        print(Matrix)
    else:
        Matrix = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        print(Matrix)



#TestGetStrurture(2)



def TestCreateStrurture():
    kernel_one=np.ones((5,5),np.uint)
    print(kernel_one)
    kernel_two=np.array([[2,3,1,3,5],[3,2,3,52,5],[2,1,423,5,3],[4,2,32,53,3],[3,5,56,2,31]],dtype=np.uint8)
    print(kernel_two)



TestCreateStrurture()