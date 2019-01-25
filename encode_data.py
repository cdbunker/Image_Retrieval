# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:56:57 2018
Could make this a lot faster by doing it in batches
@author: Colin
"""

import numpy as np

def encode_data(x_train, x_test, model, out_dim):

    x_train_encoded = np.zeros((len(x_train), out_dim))
    
    for (i,image) in zip(range(len(x_train)),x_train):
        image = np.expand_dims(image,0)
        image = np.expand_dims(image,3)
        encoded = model.predict(image)
    
        x_train_encoded[i,:] = encoded
        
        if not i%500:
            print(i)
            
    x_test_encoded = np.zeros((len(x_test), out_dim))
    
    for (i,image) in zip(range(len(x_test)),x_test):
        image = np.expand_dims(image,0)
        image = np.expand_dims(image,3)
        encoded = model.predict(image)
    
        x_test_encoded[i,:] = encoded
        
        if not i%500:
            print(i)
    
    return x_train_encoded, x_test_encoded
