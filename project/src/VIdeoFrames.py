# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 06:51:21 2018

@author: DELL
"""


import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
      vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
      success,image = vidcap.read()
      print ('Read a new frame: ', success)
      if(success):
          cv2.imwrite( pathOut + "\\frame_00%d.jpg" % count, image)     # save frame as JPEG file
      count += 1
    
    