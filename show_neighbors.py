# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:59:06 2018

@author: Colin
"""

import matplotlib.pyplot as plt
import numpy as np

def show_neighbors(train, test_flat, test_ims, y, num_retrievals, neigh):

    
    fig, ax = plt.subplots(10, num_retrievals+1)
    
    for i in range(10):
        ind = np.where(y==i)[0][2]
        im = test_flat[ind,:]
        im = np.expand_dims(im,0)
        show_im = test_ims[ind,:,:]
        ax[i,0].imshow(show_im)
        ax[i,0].axis('off')
        n = neigh.kneighbors(im)[1][0]
                
        for j in range(len(n)):
            show_im = train[n[j],:,:]
            ax[i,j+1].imshow(show_im)
            ax[i,j+1].axis('off')
            