"""
Created on Wed Dec  1 17:08:42 2021

@author: can
"""

import numpy as np
from sklearn.feature_extraction import image

def get_patches_sparse(np_training_input,window_size,im_size):
    width=im_size[0]
    height=im_size[1]
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
                
    
    return x,y


def patch_pruning(Xh, Xl):
    pvars = np.var(Xh, axis=0)
    threshold = np.percentile(pvars, 10)
    idx = pvars > threshold
    # print(pvars)
    Xh = Xh[:, idx]
    Xl = Xl[:, idx]
    return Xh, Xl