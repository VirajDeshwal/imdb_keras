#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:45:37 2018

@author: virajdeshwal
"""

'''We will be Analyzing the IMDB of movie data from the IMDB website'''

import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer


'''Let's load the data'''
(x_train, y_train), (x_test, y_test ) = imdb.load_data(num_words=1000)

'''LET'S PRE-PROCESS THE DATA'''
#lets check the shape of the words.
'''
1 comes as a positive sentiment and 0 comes as a negetive sentiment. The Dataset is pre-processed
and all the sentiments are indexed labeled.'''
print(x_train.shape)
print(y_train.shape)
'''So if we check the index of word . It will show us the index of that word in the whole corpus'''
print(x_train[0])
print(y_train[0])

'''We will do the One-hot encoding for the labels'''

#one hot-encoding the done to change the index into the vector form.

tokenizer =Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode = 'binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode = 'binary')


#One-hot the output as well. I mean Y dataset

num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)

'''Preprocessing is done. Now Let's build our own model.'''


#model sequebtial

model = Sequential()
model.add(Dense(512, activation = 'relu', input_dim = 1000))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))
model.summary()

'''Model is build ... Time to compile the model to make it ready for use.'''

model.compile(optimizer = 'rmsprop', loss= 'categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), verbose =2)