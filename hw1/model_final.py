
# coding: utf-8

# In[44]:

import time
start_time = time.time()
print ('strating time is {}'.format(start_time))
import preprocessing_2

import os
import sys
import pickle
import numpy as np
import pandas as pd
import keras
import h5py

from keras.utils import to_categorical
# from keras.layers import GRU, LSTM, Dropout, Dense, Input, TimeDistributed, Activation, Flatten, Concatenate
from keras.layers import *
from keras.models import Model, Sequential
from keras.models import load_model
# from keras.callbacks import *

import tensorflow as tf

def init():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)

init()


# In[46]:

n_user_train = 462
n_user_test = 74
n_sen_train = 1716
n_sen_test = 342

test_only = 0

# path_data = sys.argv[1]
# path_output = sys.argv[2]

path_data = 'data/'
# path_output = 

mfcc_or_fbank = 'mfcc'
model_name = 'CNN' # CNN or RNN
GL = 'GRU' # GRU or LSTM 

if mfcc_or_fbank == 'mfcc' :
    dim = 39
else :
    dim = 69

n_seq = 33
n_CNN_window = 3

# for traingin 
batch_size = 1024

# for pred_to_ans
size_window = 9


# In[ ]:




# In[8]:

#
# RNN model
#
def RNN_model() :
    dr_r = 0.5
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

    if GL == 'LSTM' :
        B1 = wrappers.Bidirectional(LSTM(64, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(I)
        B2 = wrappers.Bidirectional(LSTM(128, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(B1)
        B3 = wrappers.Bidirectional(LSTM(128, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(B2)
        B4 = wrappers.Bidirectional(LSTM(64, activation='elu', dropout=dr_r, return_sequences=True), merge_mode='concat', weights=None)(B3)
    elif GL == 'GRU' :
        #B1 = wrappers.Bidirectional(GRU(39, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(I)
        #B2 = wrappers.Bidirectional(GRU(39, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(B1)
        #B4 = wrappers.Bidirectional(GRU(78, activation='elu', dropout=dr_r, return_sequences=True), merge_mode='concat', weights=None)(B2)
        
        B1 = wrappers.Bidirectional(GRU(39, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(I)
        B4 = wrappers.Bidirectional(GRU(96, activation='elu', dropout=dr_r, return_sequences=True), merge_mode='concat', weights=None)(B1)
        
        #B1 = wrappers.Bidirectional(GRU(64, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(I)
        #B2 = wrappers.Bidirectional(GRU(128, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(B1)
        #B3 = wrappers.Bidirectional(GRU(128, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(B2)
        #B4 = wrappers.Bidirectional(GRU(64, activation='elu', dropout=dr_r, return_sequences=True), merge_mode='concat', weights=None)(B2)
        
        
    #gru100 = GRU(48, activation='softmax', dropout=0.0, return_sequences=True)(B4)
    gru100 = wrappers.Bidirectional(GRU(48, activation='softmax', dropout=0.0, return_sequences=True), merge_mode='ave', weights=None)(B4)

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


# In[ ]:

#
# CNN model
#
def CNN_model() :
    dr_r = 0.5
    if mfcc_or_fbank == 'mfcc' :
        I = Input(shape=((n_seq,n_CNN_window,int(dim/3),3))) # shape = (?,1,200,2)
    if GL == 'LSTM' :
        B1 = wrappers.Bidirectional(LSTM(64, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(I)
        B2 = wrappers.Bidirectional(LSTM(128, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(B1)
        B3 = wrappers.Bidirectional(LSTM(128, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(B2)
        B4 = wrappers.Bidirectional(LSTM(64, activation='elu', dropout=dr_r, return_sequences=True), merge_mode='concat', weights=None)(B3)
    elif GL == 'GRU' :
        T1 = TimeDistributed(Conv2D(64,(3,1), strides=(1,1), activation='elu'))(I)
        T2 = TimeDistributed(Conv2D(64,(1,3), strides=(1,1), activation='elu'))(T1)
        TM1 = TimeDistributed(MaxPooling2D(pool_size=(1, 2), strides=(1,1), padding='valid'))(T2)
        T3 = TimeDistributed(Flatten())(TM1)
        B1 = wrappers.Bidirectional(GRU(64, activation='elu', dropout=0.0, return_sequences=True), merge_mode='concat', weights=None)(T3)
        B2 = wrappers.Bidirectional(GRU(64, activation='elu', dropout=dr_r, return_sequences=True), merge_mode='concat', weights=None)(B1)
        B4 = wrappers.Bidirectional(GRU(128, activation='elu', dropout=dr_r, return_sequences=True), merge_mode='concat', weights=None)(B2)
        
    gru100 = wrappers.Bidirectional(GRU(48, activation='softmax', dropout=0.0, return_sequences=True), merge_mode='ave')(B4)

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


# In[ ]:

def predict_to_ans(ary_pred, model_name, mfcc_or_fbank, n_seq, GL, size_window, n_CNN_window) :
    def num_to_char(ary_pred_num) :
        map_48phone_char = pd.read_csv('{}48phone_char.map'.format(path_data), header=None, delimiter='\t')
        dict_map_48phone_char = dict()
        for row in map_48phone_char.iterrows() :
            #dict_map_48char[name[1][0]] = dict_map_39char[name[1][1]]
            dict_map_48phone_char[row[1][1]] = row[1][2]
        flat = ary_pred_num.flatten()
        flat_copy = flat.copy().astype(str)
        for i,num in enumerate(flat) :
            flat_copy[i] = dict_map_48phone_char[num]
        ary_pred_char = flat_copy.reshape((-1,n_seq))
        return ary_pred_char

    def find_mass_row(string) :
        dict_count = dict()
        for item in string :
            try :
                dict_count[item] += 1
            except :
                dict_count[item] = 1
        max_item = max(dict_count, key=dict_count.get)
        return max_item
    
    def find_mass_column(ary_pred, i_data_total) :
        dict_count = dict()
        for i in range(n_seq) :
            item = ary_pred[i_data_total-n_seq+1+i][n_seq-1-i]
            try :
                dict_count[item] += 1
            except :
                dict_count[item] = 1
        max_item = max(dict_count, key=dict_count.get)
        return max_item
    
    def to_str_final(string) :
        #
        # combine two char if they are same
        #
        str_final = string[0]
        for c in string[1:] :
            if c != str_final[-1] :
                str_final += c
        #
        # cut all sil in the begining
        #
        if str_final[0] == 'L' :
            str_final = str_final[1:]
            
        if str_final[-1] == 'L' :
            str_final = str_final[:-1]
            
        return str_final
    
    df_BE_test = pd.read_csv('./data_pp/beginEnd_test.csv')
    
    ary_pred_num = np.argmax(ary_pred, axis=2)
    #print (ary_pred_num[200:206])
    ary_pred_char = num_to_char(ary_pred_num)
    
    lst_X_data = []
    i_data_total = 0 # for loop use
    ans = []
    for BE in df_BE_test.iterrows() :
        index_begin = BE[1]['index_begin']
        index_end = BE[1]['index_end']
        length_BE = BE[1]['length']
        if model_name == 'RNN' :
            n_data = length_BE - n_seq + 1
        elif model_name == 'CNN' :
            n_data = length_BE - n_seq + 1 - n_CNN_window + 1
        assert n_data >= 1, 'n_data should bigger than 1, please do checking'
        
        str_temp = ''
        str_temp_2 = ''
        for i_data in range(n_data) :
            dic_temp = dict()
            if (i_data < n_seq - 1) : # don't count the begining and end sequence
                i_data_total += 1
                continue
            else :
                str_temp += str(find_mass_column(ary_pred_char, i_data_total))
                i_data_total += 1
        assert len(str_temp) == n_data - n_seq + 1, 'len(str_temp) != n_data - n_seq + 1, please check'
        
        for t in range(len(str_temp) - size_window + 1) :
            str_temp_2 += str(find_mass_row(str_temp[t:t+size_window]))
        assert len(str_temp_2) == len(str_temp) - size_window + 1, 'len(str_temp_2) != len(str_temp) - size_window + 1, please check'
        
        str_final = to_str_final(str_temp_2)
        
        ans += [str_final]
        
    print ('max_len_ans : {}'.format(max(len(x) for x in ans)))
    
    sample = pd.read_csv('{}sample.csv'.format(path_data))
    assert len(sample['phone_sequence']) == len(ans), 'len(sample[\'phone_sequence\']) != len(ans), please check'
    
    sample['phone_sequence'] = pd.DataFrame(ans)
    if not os.path.isdir('./ans') :
        os.mkdir('./ans')
    sample.to_csv('./ans/ans_{}_{}_{}_{}_WS{}.csv'.format(model_name, mfcc_or_fbank, n_seq, GL, size_window), index=False)
        
    return sample


# In[12]:

def do_training() :
    #
    # loading data
    #
    X_train = np.load('./data_pp/X_train_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq))
    y_train = np.load('./data_pp/y_train_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq))
    
    y_train = y_train.reshape((-1))
    
    if model_name == 'RNN' :
        y_train_dummy = to_categorical(y_train, num_classes=48)
        y_train_dummy = y_train_dummy.reshape((-1,n_seq,48))
    
    if model_name == 'CNN' :
        X_train = X_train.reshape((-1,n_seq,n_CNN_window,int(dim/3),3))
        y_train = y_train.reshape((-1,n_seq,n_CNN_window))
        y_train_dummy = to_categorical(y_train, num_classes=48)
        y_train_dummy = y_train_dummy.reshape((-1,n_seq,48))
        print ('X_train.shape : ')
        print (X_train.shape)
        print ('y_train_dummy.shape : ')
        print (y_train_dummy.shape)
    
    if not os.path.isdir('./model') :
        os.mkdir('./model')

    MCP = keras.callbacks.ModelCheckpoint('./model/{}_{}_{}_{}.h5'.format(model_name, mfcc_or_fbank, n_seq, GL), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    ES = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=0, mode='auto')
    #RM = keras.callbacks.RemoteMonitor(root='http://localhost:12333', path='/publish/epoch/end/', field='data', headers=None)

    if model_name == 'RNN' :
        model = RNN_model()
    elif model_name == 'CNN' :
        model = CNN_model()
    model.fit(X_train, y_train_dummy, epochs=200, batch_size=batch_size, validation_split=0.1, callbacks=[MCP,ES])



# In[ ]:

def do_testing(lst_size_window, n_CNN_window) :
    #
    # loading data
    #
    X_test = np.load('./data_pp/X_test_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq))
    model = load_model('./model/{}_{}_{}_{}.h5'.format(model_name, mfcc_or_fbank, n_seq, GL))
    
    if model_name == 'CNN' :
        X_test = X_test.reshape((-1,n_seq,n_CNN_window,int(dim/3),3))
        print ('X_test.shape : ')
        print (X_test.shape)

    pred = model.predict(X_test)
    for size_window in lst_size_window :
        ans = predict_to_ans(pred, model_name, mfcc_or_fbank, n_seq, GL, size_window, n_CNN_window)
        print (ans[:5])


# In[20]:

lst_size_window = [7]
lst_n_seq = [13]

for n_seq in lst_n_seq :
#     if model_name == 'RNN' :
#         preprocessing.preprocessing(path_data,model_name,mfcc_or_fbank,n_seq,n_CNN_window)
#     elif model_name == 'CNN' :
    if test_only :
        preprocessing_2.preprocessing_test_only(path_data,model_name,mfcc_or_fbank,n_seq,n_CNN_window)
    else :
        preprocessing_2.preprocessing(path_data,model_name,mfcc_or_fbank,n_seq,n_CNN_window)
        do_training()
    do_testing(lst_size_window, n_CNN_window)
    
print ("My program took", str(time.time() - start_time), "to run")


# In[ ]:


    


# In[ ]:




# In[ ]:



