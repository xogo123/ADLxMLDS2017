
# coding: utf-8

# In[1]:

import sys
import pickle
import numpy as np
import pandas as pd
import keras

from keras.utils import to_categorical
# from keras.layers import GRU, LSTM, Dropout, Dense, Input, TimeDistributed, Activation, Flatten, Concatenate
from keras.layers import *
from keras.models import Model, Sequential

import tensorflow as tf

def init():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)

init()


# In[2]:

n_user_train = 462
n_user_test = 74
n_sen_train = 1716
n_sen_test = 342

path_data = 'data/'
mfcc_or_fbank = 'mfcc'
RNN_or_RCNN = 'RNN'

if mfcc_or_fbank == 'mfcc' :
    dim = 39
else :
    dim = 69

# RNN seting
n_RNN_seq = 3

if RNN_or_RCNN == 'RNN' :
    n_seq = n_RNN_seq
    
# for traingin 
batch_size = 1024


# In[8]:

#
# RNN model
#
def RNN_model() :
    dr_r = 0.25
    I = Input(shape=((n_seq,dim))) # shape = (?,1,200,2)
#     gru1 = GRU(32, activation='relu', dropout=0.0, return_sequences=True)(I)
#     gru2 = GRU(32, activation='relu', dropout=0.0, return_sequences=True)(gru1)
#     gru3 = GRU(64, activation='relu', dropout=0.0, return_sequences=True)(gru2)
#     gru4 = GRU(64, activation='relu', dropout=0.0, return_sequences=True)(gru3)
    
#     gru12 = GRU(32, activation='relu', dropout=0.0, return_sequences=True, go_backwards=True)(I)
#     gru22 = GRU(32, activation='relu', dropout=0.0, return_sequences=True, go_backwards=True)(gru12)
#     gru32 = GRU(64, activation='relu', dropout=0.0, return_sequences=True, go_backwards=True)(gru22)
#     gru42 = GRU(64, activation='relu', dropout=0.0, return_sequences=True, go_backwards=True)(gru32)
    
#     F1 = Flatten()(gru4)
#     F2 = Flatten()(gru42)
#     C1 = Concatenate()([F1,F2])
#     Dr1 = Dropout(dr_r)(C1)
    B1 = wrappers.Bidirectional(GRU(64, activation='relu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(I)
    #B2 = wrappers.Bidirectional(GRU(64, activation='relu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(B1)
    #B3 = wrappers.Bidirectional(GRU(64, activation='relu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(B2)
    B4 = wrappers.Bidirectional(GRU(64, activation='relu', dropout=dr_r, return_sequences=True), merge_mode='concat', weights=None)(B1)
    gru100 = GRU(48, activation='softmax', dropout=0.0, return_sequences=True)(B4)

    model = Model(I,gru100)
    model.compile(#loss='mean_squared_error',
                      loss='categorical_crossentropy',
                      #loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      #optimizer='adam',
                      #optimizer='sgd',
                      metrics=['acc']) #'mae'
    print (model.summary())
    return model


# In[9]:

#
# loading data
#
'{}data_pp/X_train_{}_{}_{}.npy'.format(path_data, model_name, mfcc_or_fbank, n_seq)
X_train = np.load('{}data_pp/X_train_{}_{}_{}.npy'.format(path_data, model_name, mfcc_or_fbank, n_seq))
y_train = np.load('{}data_pp/y_train_{}_{}_{}.npy'.format(path_data, model_name, mfcc_or_fbank, n_seq))
X_test = np.load('{}data_pp/X_test_{}_{}_{}.npy'.format(path_data, model_name, mfcc_or_fbank, n_seq))


# In[10]:




# In[11]:

y_train = y_train.reshape((-1))
y_train_dummy = to_categorical(y_train, num_classes=48)
y_train_dummy = y_train_dummy.reshape((-1,3,48))
print (X_train.shape)
print (y_train_dummy.shape)


# In[12]:

model = RNN_model()
model.fit(X_train, y_train_dummy, epochs=1, batch_size=batch_size, validation_split=0.25)
pred = model.predict(X_test)


# In[14]:

print (pred.shape)


# In[ ]:



