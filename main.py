import numpy as np
import matplotlib.pyplot as plt
import cv2
from func import kernal
cap = cv2.VideoCapture('data/obj2.mp4')

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

    w,h = frame.shape[:2]
    frame = cv2.resize(frame, (h//2,w//2))
    mask_roi = cv2.resize(mask_roi, (h//2,w//2))

    cv2.imshow('frame',frame)
    cv2.imshow('ROI',mask_roi)

    if cv2.waitKey(10) == 27:
        break

cv2.release()
cv2.destroyAllWindows()

