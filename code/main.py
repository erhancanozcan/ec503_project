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
#from performance_a3 import obtain_new_features,calculate_accuracy
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
