#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 03:26:04 2021

@author: can
"""
path="/Users/can/Desktop/ec503_project/"
#%%

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
from read_data_a3 import prepare_train_test
#from plot_a3 import plot_eigen_faces,eigen_face_projection,draw_picture_with_n_features
import math
import pandas as pd
from performance_a3 import obtain_new_features,calculate_accuracy
from scipy.spatial import distance_matrix
#%%
dataset_name="ORL-DATABASE"
tr_percentage=0.8
#%%
#read data and reshape training test input
width,height,np_training_input,np_test_input,np_training_class,np_test_class=prepare_train_test(path_to_data,dataset_name,tr_percentage)
no_of_tr_pictures=len(np_training_input)
no_of_test_pictures=len(np_test_input)
#plt.imshow(np_training_input[20,],cmap="gray")
#plt.imshow(np_test_input[20,],cmap="gray")

tr_input_row=np_training_input.reshape(no_of_tr_pictures,height*width)
test_input_row=np_test_input.reshape(no_of_test_pictures,height*width)

#plt.imshow(tr_input_row.reshape(200,height,width)[20,],cmap="gray")
#plt.imshow(test_input_row.reshape(200,height,width)[20,],cmap="gray")
#%%
from disruptor import add_blur_decrease_size

desired_dim=(32,32)
width,height,np_training_input,np_test_input=add_blur_decrease_size(np_training_input,np_test_input,desired_dim)
#plt.imshow(np_training_input[20,],cmap="gray")
tr_input_row=np_training_input.reshape(no_of_tr_pictures,height*width)
test_input_row=np_test_input.reshape(no_of_test_pictures,height*width)
#%%
tr_input_col=np.transpose(tr_input_row)
test_input_col=np.transpose(test_input_row)

#to calculate average face, we need to calculate average of each row.
average_face=np.mean(tr_input_col,axis=1)
#average_face.reshape(10304,1)

difference_faces=np.transpose(np.array([average_face,]*no_of_tr_pictures))
difference_faces=tr_input_col-difference_faces

aTa=np.matmul(np.transpose(difference_faces),difference_faces)
from numpy import linalg as LA
eig_vals,eig_vecs=LA.eig(aTa)
#each column of eig_vecs is a eigen vector.

U=np.matmul(difference_faces,eig_vecs)
#U is scaled by dividing its length
length_U=LA.norm(U,2,axis=0)
U=U/length_U

#%%
whole_new_features=obtain_new_features(tr_input_col,test_input_col,U,average_face)

accuracy=calculate_accuracy(whole_new_features,200)

accuracy_list=np.zeros(200)

for cntr in range(200):
    accuracy_list[cntr]=calculate_accuracy(whole_new_features,cntr)
    

f, ax = plt.subplots(1)
ax.set_xlim([0,200])
ax.set_ylim((0.7,0.98))
#f.xlim((0,35))
ax.plot(range(200),accuracy_list,marker='o',label="Accuracy vs Dimensionality",markersize=3)
#f.legend()
ax.set_xlabel("Dimensionality")
ax.set_ylabel("Rank1 Identification Accuracy")
f.suptitle('Figure4: Accuracy vs Dimensionality', fontsize=12)
f


