# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 20:51:21 2018

@author: Colin
"""

from keras import backend as K
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
from convnet import convnet
import matplotlib.pyplot as plt
from Generator import Generator
from utils import triplet_loss
from utils import accuracy
from utils import l2Norm
from utils import euclidean_distance
from keras.layers import Input, Lambda
from keras.models import Model
from encode_data import encode_data
from show_neighbors import show_neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

in_dim = (28,28,1)
out_dim = 128

encoder = convnet(in_dim, out_dim)
    
in_a = Input(shape=in_dim)
in_p = Input(shape=in_dim)
in_n = Input(shape=in_dim)

emb_a = encoder(in_a)
emb_p = encoder(in_p)
emb_n = encoder(in_n)

positive_dist = Lambda(euclidean_distance, name='pos_dist')([emb_a, emb_p])
negative_dist = Lambda(euclidean_distance, name='neg_dist')([emb_a, emb_n])
tertiary_dist = Lambda(euclidean_distance, name='ter_dist')([emb_p, emb_n])

stacked_dists = Lambda(lambda vects: 
    K.stack(vects, axis=1), 
    name='stacked_dists')([positive_dist, negative_dist, tertiary_dist])
    
model = Model([in_a, in_p, in_n], stacked_dists, name='triple_siamese')
model.compile(optimizer=Adam(), loss=triplet_loss, metrics=[accuracy])

dg_train = Generator(x_train, y_train)
dg_test = Generator(x_test, y_test)

model.fit_generator(dg_train, epochs=10, steps_per_epoch=2000,
                    validation_data=dg_test, validation_steps=2000, workers=8)

test_model = encoder

x_train_encoded, x_test_encoded = encode_data(x_train, x_test, 
                                              test_model, out_dim)

num_retrievals = 10

x_train2 = np.reshape(x_train, (60000, 28*28))
x_test2 = np.reshape(x_test, (10000, 28*28))
neigh1 = NearestNeighbors(n_neighbors=num_retrievals)
neigh1.fit(x_train2) 
show_neighbors(x_train, x_test2, x_test, y_test, num_retrievals, neigh1)

neigh2 = NearestNeighbors(n_neighbors=num_retrievals)
neigh2.fit(x_train_encoded) 
show_neighbors(x_train, x_test_encoded, x_test, y_test, num_retrievals, neigh2)

orig_embedded = TSNE(n_components=2).fit_transform(x_test2)
encoded_embedded = TSNE(n_components=2).fit_transform(x_test_encoded)

fig, ax = plt.subplots()
colors = 'r', 'g', 'm', 'c', 'b', 'y', 'k', 'w', 'orange', 'purple'
for i,color in zip(range(10), colors):
    i_ind = np.where(y_test==i)[0]
    plt.scatter(orig_embedded[i_ind,0], orig_embedded[i_ind,1], c=color, 
                label=i, edgecolors='k')
plt.legend()
    
    
fig, ax = plt.subplots()
for i,color in zip(range(10), colors):
    i_ind = np.where(y_test==i)[0]
    plt.scatter(encoded_embedded[i_ind,0], encoded_embedded[i_ind,1], c=color,
                label=i, edgecolors='k')
plt.legend()
    