import numpy as np
import matplotlib.pyplot as plt
import cv2
from func import kernal, max_area_idx
cap = cv2.VideoCapture('data/obj2.mp4')
y=240
x1=560
x2=600
count=0
k=0
m=0
while True:
    ret,frame = cap.read()
    # cv2.imwrite('frame.png', frame)

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    zeros_mask = np.zeros(gray.shape,dtype=np.uint8)
    mask = cv2.rectangle(zeros_mask,(1080,0),(1200,1080),color=255,thickness=-1)

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    bitwise_img = cv2.bitwise_and(RGB,mask_rgb)
    ROI_HSV = cv2.cvtColor(bitwise_img,cv2.COLOR_RGB2HSV)

    lower = np.array([9,26,114],dtype=np.uint8)
    upper = np.array([78,151,255],dtype=np.uint8)

    mask_roi = cv2.inRange(ROI_HSV,lower,upper)
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernal(10,10))
    w,h = frame.shape[:2]
    frame = cv2.resize(frame, (h//2,w//2))
    mask_roi = cv2.resize(mask_roi, (h//2,w//2))
    
    # Find bbox countors
    contours, hierarchy = cv2.findContours(mask_roi,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    mx_idx = max_area_idx(contours)
    X,Y,W,H = cv2.boundingRect(contours[mx_idx])

    # print(mask_roi.shape)
    # break
    n = count
    for i in range(x1,x2):
        if mask_roi[y][i]==255:
            count+=1
            m+=1
            break
    if n==count:
        count=0
        m=0
    if count>80:
        count=0
        k+=1
        print(k)
    if m>150:
        count=0
    frame=cv2.putText(frame,f"COUNTING: {k}",org=(720,65),fontScale=1,fontFace=0,color=(0,33,255),thickness=2)
    frame = cv2.rectangle(frame,(X,Y),(X+W,Y+H),(61,255,21),1)
    
    cv2.imshow('frame',frame)

    if cv2.waitKey(33) == ord('q'):
        break

cv2.release()
cv2.destroyAllWindows()

