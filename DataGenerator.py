# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 01:28:59 2018

@author: Colin
"""

import numpy as np
import keras
import random

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, series):
        'Initialization'
        self.series = series
        
    def __len__(self):
      'Denotes the number of batches per epoch'
      return 100
        
    def on_epoch_end(self):
        return 

    def __getitem__(self, index):
        r = random.sample(range(0, 568), 32)
        q = random.randint(0,12)
        q2 = q+64
        
        s = self.series[q:q2,r].T
        s = np.expand_dims(s,2)
        
        return s, s