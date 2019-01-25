# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:30:50 2018

@author: Colin
"""
from keras import backend as K

def triplet_loss(y_true, y_pred):
    margin = K.constant(2)
    return K.mean(K.maximum(
            K.constant(0),
            K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))

def accuracy(y_true, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])

def l2Norm(x):
    return  K.l2_normalize(x, axis=-1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y),
                                  axis=1, keepdims=True), K.epsilon()))