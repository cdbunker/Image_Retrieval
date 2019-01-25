# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:30:49 2018

@author: Colin
"""
from keras.layers import Input, Conv2D, Activation, Dense, Flatten, Lambda
from keras.layers import BatchNormalization
from keras.models import Model, Sequential
from keras import backend as K

def convnet(in_dim, out_dim):
    encoder = Sequential()
    encoder.add(BatchNormalization(input_shape=in_dim))
    encoder.add(Conv2D(64, (3, 3), padding='same'))
    encoder.add(Conv2D(64, (1,1), strides=(2,2), padding='same'))
    encoder.add(BatchNormalization())
    encoder.add(Activation('relu'))
    encoder.add(Conv2D(64, (3, 3), padding='same'))
    encoder.add(Conv2D(64, (1,1), strides=(2,2), padding='same'))
    encoder.add(BatchNormalization())
    encoder.add(Activation('relu'))
    encoder.add(Conv2D(64, (3, 3), padding='same'))
    encoder.add(BatchNormalization())
    encoder.add(Activation('relu'))
    encoder.add(Flatten())
    encoder.add(Dense(out_dim, activation='linear'))
    encoder.add(Lambda(lambda  x: K.l2_normalize(x,axis=1), name='norm'))
    
    return encoder