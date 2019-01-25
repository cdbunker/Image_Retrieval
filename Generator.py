# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 01:26:43 2018

@author: Colin
"""
import keras
import random
import numpy as np

class Generator(keras.utils.Sequence):
    def __init__(self, data, labels):
        'Initialization'
        self.data = data
        self.labels = labels
        
    def __len__(self):
      'Denotes the number of batches per epoch'
      return 100
        
    def on_epoch_end(self):
        return 

    def __getitem__(self, index):
        q = random.randint(0,len(self.labels)-1)
        num = self.labels[q]
        similar = np.where(self.labels==num)[0]
        dissimilar = np.where(self.labels!=num)[0]
        
        similar_q = random.randint(0,len(similar)-1)
        similar_data_index = similar[similar_q]
        dissimilar_q = random.randint(0,len(dissimilar)-1)
        dissimilar_data_index = dissimilar[dissimilar_q]
        
        anchor = np.expand_dims(self.data[q,:,:],0)
        anchor = np.expand_dims(anchor,3)
        positive = np.expand_dims(self.data[similar_data_index,:,:],0)
        positive = np.expand_dims(positive,3)
        negative = np.expand_dims(self.data[dissimilar_data_index,:,:],0)
        negative = np.expand_dims(negative,3)
        
        return [anchor, positive, negative], np.zeros(1)