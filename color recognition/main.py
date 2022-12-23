# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 19:15:24 2022

@author: safak
"""

import cv2
import numpy as np
from collections import deque

# datatype that stores object's center
buffer_size=32
pts=deque(maxlen=buffer_size)

# green color range
gr_lower=(60,120,50)
gr_upper=(90,255,255)

# capture processing
cap=cv2.VideoCapture(0)
cap.set(3,960)
cap.set(4,480)

while True:
    ret,frame_original=cap.read()
    
    if ret:
        # blurring
        blurred_img=cv2.GaussianBlur(frame_original, (11,11), 0)
        # convert to HSV
        hsv_img=cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
        # create a mask for green color
        mask=cv2.inRange(hsv_img, gr_lower, gr_upper)
        # erode and dilate processing for destroy the noises
        mask=cv2.erode(mask,None,iterations=2)
        mask=cv2.dilate(mask,None,iterations=2)
        #cv2.imshow("image with mask",mask)
        
        # contours 
        (contours,_)=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center=None
        
        if len(contours) > 0:
            # get the biggest contour
            con=max(contours,key=cv2.contourArea)
            # convert to rectangle
            rect=cv2.minAreaRect(con)
            
            ((x,y),(width,height),rotation)=rect

            text="x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x),np.round(y),np.round(width),
                                                                            np.round(height),np.round(rotation))
            # create the box
            box=cv2.boxPoints(rect)
            box=np.int64(box)
            
            # moment (get the center point of box/contour area)
            mom=cv2.moments(con)
            center=(int(mom["m10"]/mom["m00"]),int(mom["m01"]/mom["m00"]))
            
            # draw contour
            cv2.drawContours(frame_original, [box], 0, (0,0,255),1)
            # draw a point on the center
            cv2.circle(frame_original,center,5,(255,0,0),-1)
            # draw the informations
            cv2.putText(frame_original,text,(25,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            
        # deque (stores the points of last 32 centers)
        pts.appendleft(center)
        
        for i in range(1,len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            cv2.line(frame_original,pts[i-1],pts[i],(255,0,0),2)
            
        cv2.imshow("original detection",frame_original)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break








cap.release()
cv2.destroyAllWindows()















