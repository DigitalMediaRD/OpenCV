import  cv2
img1=cv2.imread('res/img001.png',cv2.IMREAD_REDUCED_COLOR_2)
img2=cv2.imread('res/img002.png',cv2.IMREAD_REDUCED_COLOR_2)

def AddWeight():
    img=cv2.addWeighted(img1,0.9,img2,0.3,30)
    cv2.imshow('Weight',img)
    cv2.waitKey(0)
    
    
    
AddWeight()