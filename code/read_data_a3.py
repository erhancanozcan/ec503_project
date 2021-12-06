#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 03:00:04 2021

@author: can
"""

#%%
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
import cv2
from PIL import Image
#%%
def prepare_train_test(path_to_data,dataset_name,tr_percentage):
    if dataset_name=="ORL-DATABASE":
        path_to_data=path_to_data+"/"+dataset_name
        width=92
        height=112
        no_of_training=round(400*tr_percentage)
        no_of_test=round(400*(1-tr_percentage))
        
        training_input=[None]*no_of_training
        training_class=[None]*no_of_training
        test_input=[None]*no_of_test
        test_class=[None]*no_of_test
        
        instances = glob.glob(path_to_data+"//*")
        tr_counter=0
        test_counter=0
        
        for inst in instances:
            if inst[-3:]!="TXT":
                pers_no=inst[-3:]
                if pers_no[0]=="s":
                    pers_no=int(pers_no[-2:])
                else:
                    pers_no=int(pers_no[-1:])
            
            sessions=glob.glob("%s//*" %inst)
            for ses in sessions:
                ses_no=ses[-6:]
                if ses_no[0]=="/":
                    ses_no=int(ses_no[1])
                else:
                    ses_no=int(ses_no[:2]) 
                

                if ses_no <= round(10*tr_percentage):
                    training_input[tr_counter]=cv2.imread(ses,cv2.IMREAD_GRAYSCALE)
                    training_class[tr_counter]=pers_no
                    tr_counter=tr_counter+1
                else:
                    test_input[test_counter]=cv2.imread(ses,cv2.IMREAD_GRAYSCALE)
                    test_class[test_counter]=pers_no
                    test_counter=test_counter+1        
      
    
        """   
        plt.imshow(training_input[0],cmap="gray")
        plt.imshow(training_input[1],cmap="gray") 
        plt.imshow(training_input[2],cmap="gray") 
        plt.imshow(training_input[3],cmap="gray") 
        plt.imshow(training_input[4],cmap="gray")  
        plt.imshow(test_input[0],cmap="gray")
        plt.imshow(test_input[1],cmap="gray") 
        plt.imshow(test_input[2],cmap="gray") 
        plt.imshow(test_input[3],cmap="gray") 
        plt.imshow(test_input[4],cmap="gray")  
        """
        np_training_input=np.array(training_input)
        np_test_input=np.array(test_input)
        np_training_class=np.array(training_class)
        np_test_class=np.array(test_class)
        #plt.imshow(np_training_input[2,],cmap="gray")
        return width,height,np_training_input,np_test_input,np_training_class,np_test_class

#%%
def prepare_images_att(num_people,num_pic_tr,num_pic_te):
    path = '/Users/can/Desktop/ec503_project/data/ORL-DATABASE'
    X = []
    y = []
    for i in range(40):
        for j in range(10):
            img = Image.open(os.path.join(path +'/s'+str(i+1), str(j+1)+'.pgm'))
            X.append(np.asarray(img, dtype=np.uint8).flatten())
            y.append(i)
    X = np.asarray(X) #All images
    y = np.asarray(y) #numbered people (40 in total)



    #Create the training and testing datasets
    width = 92
    height = 112
    
    
    X=X.reshape(400,height,width)

    
    
    
    
    
    ind=np.arange(num_people)
    train_ind=ind*10
    tmp=ind*10
    
    for i in range(num_pic_tr-1):
        train_ind=np.concatenate([train_ind,tmp+1+i])
        
    train_ind=np.sort(train_ind)
    
    np_training_input=X[train_ind,]    
    np_training_class=y[train_ind,]
    
    
    ind=np.arange(num_people)
    test_ind=ind*10+num_pic_tr
    tmp=ind*10+num_pic_tr
    for i in range(num_pic_te-1):
        test_ind=np.concatenate([test_ind,tmp+1+i])
        
    test_ind=np.sort(test_ind)
    
    np_test_input=X[test_ind,]    
    np_test_class=y[test_ind,]
    



    return width,height,np_training_input,np_test_input,np_training_class,np_test_class
    