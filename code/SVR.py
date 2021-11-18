#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 23:26:44 2021

@author: can
"""

from sklearn.feature_extraction import image
import numpy as np


def get_patches(np_training_input,desired_dim,window_size=(5,5)):
    
    width=desired_dim[0]
    height=desired_dim[1]
    tmp=int(window_size[0]-1)
    one_tmp=int(tmp/2)

    y=np.zeros(width*height*len(np_training_input))
    x=np.zeros((width*height*len(np_training_input),window_size[0]*window_size[1]))


    for z in range (len(np_training_input)):
        array_tmp=z*len(np_training_input)
        img=np_training_input[z,]
        zimg = np.zeros((height+tmp, width+tmp))
        zimg[one_tmp:height+one_tmp, one_tmp:width+one_tmp] = img
        
        patches = image.extract_patches_2d(zimg, window_size)
        
        
        
        #y=np.zeros(len(patches))
        #x=np.zeros((len(patches),window_size[0]*window_size[1]))
        for i in range (height):
            for j in range (width):
                y[array_tmp+i*j+j]=img[i,j]
                x[array_tmp+i*j+j]=patches[i*j+j].flatten()
                
    
    return(x,y)

def get_patches_single_image(np_training_input,desired_dim,window_size=(5,5)):
    
    width=desired_dim[0]
    height=desired_dim[1]
    tmp=int(window_size[0]-1)
    one_tmp=int(tmp/2)

    y=np.zeros(width*height)
    x=np.zeros((width*height,window_size[0]*window_size[1]))


    
    
    img=np_training_input
    zimg = np.zeros((height+tmp, width+tmp))
    zimg[one_tmp:height+one_tmp, one_tmp:width+one_tmp] = img
    
    patches = image.extract_patches_2d(zimg, window_size)
    
    
    
    #y=np.zeros(len(patches))
    #x=np.zeros((len(patches),window_size[0]*window_size[1]))
    for i in range (height):
        for j in range (width):
            #y[i*j+j]=img[i,j]
            x[i*j+j]=patches[i*j+j].flatten()
                
    y=img.flatten()
    return(x,y)