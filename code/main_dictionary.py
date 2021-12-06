#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:07:22 2021

@author: can
"""

path="/Users/can/Desktop/ec503_project/"
#%%
from os import listdir
import os
#path_to_data="/Users/can/Desktop/ec503_project/ORL-DATABASE"
path_to_data=path+"data"
#path_to_non_human="/Users/can/Documents/Biometrics/a3/non_human.jpeg"
os.chdir(path+"code")
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline
import cv2
from read_data_a3 import prepare_train_test,prepare_images_att
#from plot_a3 import plot_eigen_faces,eigen_face_projection,draw_picture_with_n_features
import math
import pandas as pd
from performance_a3 import obtain_new_features,calculate_accuracy
from scipy.spatial import distance_matrix
from identifier import eigen_face
#from skimage.util.shape import view_as_windows
from sklearn.feature_extraction import image
#%%
dataset_name="ORL-DATABASE"
tr_percentage=0.9
#%%
#read data and reshape training test input
#width,height,np_training_input,np_test_input,np_training_class,np_test_class=prepare_train_test(path_to_data,dataset_name,tr_percentage)
num_people=20
num_tr=3
num_te=1
width,height,np_training_input,np_test_input,np_training_class,np_test_class = prepare_images_att(num_people,num_tr,num_te)



no_of_tr_pictures=len(np_training_input)
no_of_test_pictures=len(np_test_input)
#plt.imshow(np_training_input[0,],cmap="gray")
#plt.imshow(np_test_input[20,],cmap="gray")

tr_input_row=np_training_input.reshape(no_of_tr_pictures,height*width)
test_input_row=np_test_input.reshape(no_of_test_pictures,height*width)

#plt.imshow(tr_input_row.reshape(200,height,width)[20,],cmap="gray")
#plt.imshow(test_input_row.reshape(200,height,width)[20,],cmap="gray")
#accu_list=eigen_face(tr_input_row,test_input_row)
#%%
"""
from disruptor import add_blur_decrease_size

desired_dim=(32,32)
width,height,np_training_input=add_blur_decrease_size(np_training_input,desired_dim,True)
width,height,np_test_input=add_blur_decrease_size(np_test_input,desired_dim,True)
#plt.imshow(np_training_input[20,],cmap="gray")
tr_input_row=np_training_input.reshape(no_of_tr_pictures,height*width)
test_input_row=np_test_input.reshape(no_of_test_pictures,height*width)
"""
#%%
from disruptor import add_blur_decrease_size

resize_dim=(64,64)

width,height,np_training_input=add_blur_decrease_size(np_training_input,resize_dim,add_blur=False)
width,height,np_test_input=add_blur_decrease_size(np_test_input,resize_dim,add_blur=False)
#plt.imshow(np_training_input[20,],cmap="gray")
desired_dim=(32,32)

width,height,np_training_input_low_qual=add_blur_decrease_size(np_training_input,desired_dim,add_blur=False)
width,height,np_test_input_low_qual=add_blur_decrease_size(np_test_input,desired_dim,add_blur=False)
#plt.imshow(np_training_input_low_qual[20,],cmap="gray")
resize_dim=(64,64)
width,height,interpolate_tr=add_blur_decrease_size(np_training_input_low_qual,resize_dim,add_blur=False)
width,height,interpolate_te=add_blur_decrease_size(np_test_input_low_qual,resize_dim,add_blur=False)
#%%
from spams import trainDL
import pickle


patch_size  = 3            # image patch size
nSmp        = 500       # number of patches to sample
upscale     = 2           # upscaling factor
patch_num   = 2000

from sample_patches import sample_patches

for i in range(len(np_training_input)):
    if i==0:  
        H, L = sample_patches(np_training_input[i,], patch_size, patch_num, upscale)
    else:
        tmp_H, tmp_L = sample_patches(np_training_input[i,], patch_size, patch_num, upscale)
        H=np.concatenate([H,tmp_H],axis=1)
        L=np.concatenate([L,tmp_L],axis=1)
        
        
H = np.asfortranarray(H)
L = np.asfortranarray(L)
#%%
from spams import trainDL
dict_size=512
lmbd=0.1
Dh = trainDL(H, K=dict_size, lambda1=lmbd, iter=100)
Dl = trainDL(L, K=dict_size, lambda1=lmbd, iter=100)
#%%
from ScSR import ScSR
from backprojection import backprojection
lmbd=1.2
overlap=1
maxIter=100


pic_no=6

img_sr_y = ScSR(np_training_input_low_qual[pic_no,], resize_dim, 2, Dh, Dl, lmbd, overlap)
img_sr_y = backprojection(img_sr_y, np_training_input_low_qual[pic_no,], maxIter)

from plots import draw_triplet

draw_triplet(img_sr_y, np_training_input[pic_no,], interpolate_tr[pic_no,],from_training=True)




#%%
pic_no=2

img_sr_y = ScSR(np_test_input_low_qual[pic_no,], resize_dim, 2, Dh, Dl, lmbd, overlap)
img_sr_y = backprojection(img_sr_y, np_test_input_low_qual[pic_no,], maxIter)

draw_triplet(img_sr_y, np_test_input[pic_no,], interpolate_te[pic_no,],from_training=False)
#%%

rmse=sum(sum((img_sr_y-np_training_input[pic_no,])**2))/(resize_dim[0]*resize_dim[1])
rmse=sum(sum((interpolate_tr[pic_no,]-np_training_input[pic_no,])**2))/(resize_dim[0]*resize_dim[1])






