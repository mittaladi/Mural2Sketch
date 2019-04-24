# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:40:45 2019

@author: MitTal
"""

import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
from l0smooth.L0_serial import l0_smooth
#import src
#import potrace



image = cv2.imread('Dataset\\img8.png')
#image = cv2.resize(image , (200,300))
image_l0 = l0_smooth(image)
gray_img = cv2.cvtColor(image_l0 ,cv2.COLOR_BGR2GRAY)
image_blur = cv2.GaussianBlur(image_l0, (5, 5), 0)
img_can = cv2.Canny(image_blur.astype('uint8'), 1000,2000, True, apertureSize=5)
img_can_invrt = np.invert(img_can)
gray_blur = cv2.cvtColor(image_blur,cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray blur.png', gray_blur)

gray_blur_canny =  cv2.Canny(gray_blur.astype('uint8'), 1000, 2000, True, apertureSize=5)
gray_blur_canny_invrt = np.invert(gray_blur_canny)

cv2.imwrite('outer image.png' , gray_blur_canny_invrt)

khali_image = np.zeros((len(image),len(image[0])))
for i in range(0,len(khali_image)):
    for j in range(0,len(khali_image[0])):
        if(img_can_invrt[i][j] != 255):
            khali_image[i][j] = gray_blur[i][j]
        else:
            khali_image[i][j] = 255
            
cv2.imwrite('khali_image.png',khali_image)


image_I2 = np.zeros((len(image),len(image[0])))

for i in range(0,len(image_I2)):
    for j in range(0,len(image_I2[0])):

        if(gray_blur[i][j] == 0):
            delta_I0 = 255 
        else:
            delta_I0 = ((255-gray_blur[i][j])/gray_blur[i][j])*gray_img[i][j]

        image_I2[i][j] = min(255, (gray_img[i][j] + delta_I0))

cv2.imwrite('inner edge.png',image_I2)



image_I2 = image_I2.astype('uint8')

ret2,th2 = cv2.threshold(image_I2.flatten(),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#img_thresh_Gaussian = cv2.adaptiveThreshold(image_I2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

threshold_img = np.zeros((len(image),len(image[0])))
for i in range(0,len(image_I2)):
    for j in range(0,len(image_I2[0])):
        if(image_I2[i][j] >= ret2):
            threshold_img[i][j] = 255
        else:
            threshold_img[i][j] = 0
            
cv2.imwrite('threshold_image.png',threshold_img)
'''
image_I3 = khali_image + image_I2 - 255

cv2.imwrite('integrate image.png' , image_I3)
'''

invert_canny_img = np.zeros((len(image),len(image[0])))
for i in range(0,len(invert_canny_img)):
    for j in range(0,len(invert_canny_img[0])):
        if(img_can[i][j] == 255):
            invert_canny_img[i][j] = 0
        else:
            invert_canny_img[i][j] = 255
      
#integration of I1 and I2      
image_I3 = np.zeros((len(image),len(image[0])))
max_i = len(image)
max_j = len(image[0])
for i in range(0,len(image_I3)):
    for j in range(0,len(image_I3[0])):
        if(image_I2[i][j] != 255 ):
            if(i == 0):
                if(j == 0):
                    if(img_can_invrt[i][j+1] == 0 or img_can_invrt[i+1][j] == 0 
                               or img_can_invrt[i+1][j+1] == 0):
                        #print('hello1')

                        image_I3[i][j] = 0
                    else:
                        image_I3[i][j] = 255
                elif(j == max_j-1):
                    if(img_can_invrt[i][j-1] == 0 or img_can_invrt[i+1][j] == 0 
                               or img_can_invrt[i+1][j-1] == 0):
                        image_I3[i][j] = 0
                        #print('hello2')
                    else:
                        image_I3[i][j] = 255
                else:
                    if(img_can_invrt[i][j-1] == 0 or img_can_invrt[i+1][j] == 0 
                       or img_can_invrt[i+1][j-1] == 0 or img_can_invrt[i][j+1] == 0
                               or img_can_invrt[i+1][j+1] == 0):
                        image_I3[i][j] = 0
                        #print('hell3')
                    else:
                        image_I3[i][j] = 255
  
            elif(i == max_i-1):
                if(j == 0):
                    if(img_can_invrt[i][j+1] == 0 or img_can_invrt[i-1][j] == 0 
                               or img_can_invrt[i-1][j+1] == 0):
                        image_I3[i][j] = 0
                        #print('hello4')
                    else:
                        image_I3[i][j] = 255
                elif(j == max_j-1):
                    if(img_can_invrt[i][j-1] == 0 or img_can_invrt[i-1][j] == 0
                               or img_can_invrt[i-1][j-1] == 0):
                        image_I3[i][j] = 0
                        #print('hello5')
                    else:
                        image_I3[i][j] = 255
                else:
                    if(img_can_invrt[i][j-1] == 0 or img_can_invrt[i-1][j] == 0
                       or img_can_invrt[i-1][j-1] == 0 or img_can_invrt[i][j+1] == 0 
                               or img_can_invrt[i-1][j+1] == 0):
                        image_I3[i][j] = 0
                        #print('hello6')
                    else:
                        image_I3[i][j] = 255
            else:
                if(j == 0):
                    if(img_can_invrt[i-1][j] == 0 or img_can_invrt[i-1][j+1] == 0
                       or img_can_invrt[i][j+1] == 0 or img_can_invrt[i+1][j+1] == 0
                               or img_can_invrt[i+1][j] == 0):
                        image_I3[i][j] = 0
                        #print('hello7')
                    else:
                        image_I3[i][j] = 255
                elif(j == max_j-1):
                    if(img_can_invrt[i-1][j] == 0 or img_can_invrt[i-1][j-1] == 0 
                       or img_can_invrt[i][j-1] == 0 or img_can_invrt[i+1][j-1] == 0
                               or img_can_invrt[i+1][j] == 0):
                        image_I3[i][j] = 0
                        #print('hello8')
                    else:
                        image_I3[i][j] = 255
                    
                else:
                    if(img_can_invrt[i-1][j] == 0 or img_can_invrt[i-1][j-1] == 0 
                       or img_can_invrt[i][j-1] == 0 or img_can_invrt[i+1][j-1] == 0
                       or img_can_invrt[i+1][j] == 0 or img_can_invrt[i-1][j+1] == 0 
                       or img_can_invrt[i][j+1] == 0 or img_can_invrt[i+1][j+1] == 0):
                        image_I3[i][j] = 0
                        #print('hello9')
                    else:
                        image_I3[i][j] = 255
                
            
        else:
            image_I3[i][j] = 255
        
cv2.imwrite('integrate image.png' , image_I3)

image_I4 = np.zeros((len(image),len(image[0])))

for i in range(0,len(image_I4)):
    for j in range(0,len(image_I4[0])):
        if(threshold_img[i][j] == 255 and image_I3[i][j] == 255):
            image_I4[i][j] = 255
        else:
            image_I4[i][j] = 0
        
cv2.imwrite('image I4.png' , image_I4)

'''
hog = cv2.HOGDescriptor()

#hog_img = hog.compute(gray_blur)
'''