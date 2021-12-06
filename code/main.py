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
tr_percentage=0.8
#%%
#read data and reshape training test input
width,height,np_training_input,np_test_input,np_training_class,np_test_class=prepare_train_test(path_to_data,dataset_name,tr_percentage)
no_of_tr_pictures=len(np_training_input)
no_of_test_pictures=len(np_test_input)
#plt.imshow(np_training_input[9,],cmap="gray")
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
#plt.imshow(np_training_input[20,],cmap="gray")
desired_dim=(16,16)

width,height,np_training_input_low_qual=add_blur_decrease_size(np_training_input,desired_dim,add_blur=False)
width,height,np_test_input_low_qual=add_blur_decrease_size(np_test_input,desired_dim,add_blur=False)
#plt.imshow(np_training_input_low_qual[20,],cmap="gray")
resize_dim=(32,32)
width,height,np_training_input_low_qual=add_blur_decrease_size(np_training_input_low_qual,resize_dim,add_blur=False)
width,height,np_test_input_low_qual=add_blur_decrease_size(np_test_input_low_qual,resize_dim,add_blur=False)
#%%
from SVR import get_patches
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

w_size=(5,5)
how_many_pics_to_train=30
np.random.seed(0)
ind=np.arange(len(np_training_input))
np.random.shuffle(ind)
ind=ind[:how_many_pics_to_train]

#model is huge it never stops. try to train svr on single image.
x,y=get_patches(np_training_input_low_qual[ind,],np_training_input[ind,],resize_dim,w_size)
#%%
regr = make_pipeline(StandardScaler(), SVR(C=1000000.0, epsilon=1,verbose=True))
regr.fit(x, y)
#prediction=regr.predict(x)
prediction=regr.predict(x[:1024])
prediction=prediction[:1024]
prediction=prediction.reshape(height,width)

#%%
#image from training
#what we predicted
plt.imshow(prediction,cmap="gray")
#what we expect to see
plt.imshow(y[:1024].reshape(height,width),cmap="gray")

#what we have before resolution
plt.imshow(np_training_input_low_qual[ind[0],],cmap="gray")
#%%


#%%
#image from test
x,y=get_patches(np_test_input_low_qual,np_test_input,resize_dim,w_size)
prediction=regr.predict(x[:1024])
prediction=prediction.reshape(height,width)
#what we predicted
plt.imshow(prediction,cmap="gray")
#what we expect to see
plt.imshow(y[:1024].reshape(height,width),cmap="gray")

#what we have before resolution
plt.imshow(np_test_input_low_qual[0,],cmap="gray")

#%%
tr_input_row=np_training_input.reshape(no_of_tr_pictures,height*width)
test_input_row=np_test_input.reshape(no_of_test_pictures,height*width)
accu_list_high_quality=eigen_face(tr_input_row,test_input_row)



tr_input_row=np_training_input.reshape(no_of_tr_pictures,height*width)
test_input_row=np_test_input_low_qual.reshape(no_of_test_pictures,height*width)
accu_list_after_restoring_low_quality_via_interpolation=eigen_face(tr_input_row,test_input_row)

#%%
from SVR import get_patches
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

w_size=(7,7)
"""
#model is huge it never stops. try to train svr on single image.
x,y=get_patches(np_training_input[:50,],resize_dim,w_size)

regr = make_pipeline(StandardScaler(), SVR(C=100.0, epsilon=0.2))
regr.fit(x, y)
prediction=regr.predict(x)
prediction=prediction[:1024]
prediction=prediction.reshape(height,width)
plt.imshow(prediction,cmap="gray")
"""
#%%

np_test_input_SVR=np.zeros(np.shape(np_test_input_low_qual))
from SVR import get_patches_single_image
x,y=get_patches_single_image(np_test_input_low_qual[0,],resize_dim,w_size)


#By visual inspection, we can observe that training SVR on single image is not useful.
regr = make_pipeline(SVR(C=10, epsilon=1,kernel='rbf',degree=5))
regr.fit(x, y)
prediction=regr.predict(x)
prediction=prediction.reshape(height,width)
plt.imshow(prediction,cmap="gray")
#see the original image
y=y.reshape(height,width)
plt.imshow(y,cmap="gray")



#%%
f, ax = plt.subplots(1)
ax.set_xlim([0,200])
ax.set_ylim((0.4,0.98))
#f.xlim((0,35))
ax.plot(range(200),accu_list_high_quality,marker='o',label="Accuracy vs Dimensionality",markersize=3)
ax.plot(range(200),accu_list_after_restoring_low_quality_via_interpolation,marker='o',label="Accuracy vs Dimensionality",markersize=3)
f.legend(["accuracy_original_high_quality_test","accuracy_after_restoring_low_quality_via_interpolation"],loc='lower right',bbox_to_anchor=(0.87, 0.2))
ax.set_xlabel("Dimensionality")
ax.set_ylabel("Rank1 Identification Accuracy")
f.suptitle('Test Accuracy vs No of Features', fontsize=12)
f


#%%

window_size=(5,5)


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
        
        


#%%


