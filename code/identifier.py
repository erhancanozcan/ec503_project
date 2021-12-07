#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 19:28:40 2021

@author: can
"""
import numpy as np
import pandas as pd
from ec503_project.code.performance_a3 import obtain_new_features,calculate_accuracy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def eigen_face(tr_input_row,test_input_row,num_person,tr_pic_p_person,te_pic_p_person):

    no_of_tr_pictures=len(tr_input_row)
    no_of_test_pictures=len(test_input_row)
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
    
    
    whole_new_features=obtain_new_features(tr_input_col,test_input_col,U,average_face)
    
    accuracy=calculate_accuracy(whole_new_features,200,num_person,tr_pic_p_person,te_pic_p_person)
    
    accuracy_list=np.zeros(200)
    
    for cntr in range(200):
        accuracy_list[cntr]=calculate_accuracy(whole_new_features,cntr,num_person,tr_pic_p_person,te_pic_p_person)
        
    
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
    return(accuracy_list)
