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
from identifier import eigen_face
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

resize_dim=(32,32)

width,height,np_training_input=add_blur_decrease_size(np_training_input,resize_dim,add_blur=False)
width,height,np_test_input=add_blur_decrease_size(np_test_input,resize_dim,add_blur=False)

desired_dim=(16,16)

width,height,np_training_input_low_qual=add_blur_decrease_size(np_training_input,resize_dim,add_blur=True)
width,height,np_test_input_low_qual=add_blur_decrease_size(np_test_input,resize_dim,add_blur=True)

resize_dim=(32,32)
width,height,np_training_input_low_qual=add_blur_decrease_size(np_training_input_low_qual,resize_dim,add_blur=True)
width,height,np_test_input_low_qual=add_blur_decrease_size(np_test_input_low_qual,resize_dim,add_blur=True)

#%%
tr_input_row=np_training_input.reshape(no_of_tr_pictures,height*width)
test_input_row=np_test_input.reshape(no_of_test_pictures,height*width)
accu_list_high_quality=eigen_face(tr_input_row,test_input_row)



tr_input_row=np_training_input.reshape(no_of_tr_pictures,height*width)
test_input_row=np_test_input_low_qual.reshape(no_of_test_pictures,height*width)
accu_list_after_restoring_low_quality_via_interpolation=eigen_face(tr_input_row,test_input_row)

#%%
f, ax = plt.subplots(1)
ax.set_xlim([0,200])
ax.set_ylim((0.4,0.98))
#f.xlim((0,35))
ax.plot(range(200),accu_list_high_quality,marker='o',label="Accuracy vs Dimensionality",markersize=3)
ax.plot(range(200),accu_list_after_restoring_low_quality_via_interpolation,marker='o',label="Accuracy vs Dimensionality",markersize=3)
f.legend(["accuracy_high_quality","accuracy_after_restoring_low_quality_via_interpolation"],loc='lower right',bbox_to_anchor=(0.87, 0.2))
ax.set_xlabel("Dimensionality")
ax.set_ylabel("Rank1 Identification Accuracy")
f.suptitle('Accuracy vs Dimensionality', fontsize=12)
f


#%%


