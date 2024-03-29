
# coding: utf-8

# In[1]:


#%matplotlib inline

import time
start_time = time.time()
print ('strating time is {}'.format(start_time))
import preprocessing

import os
import sys
import pickle
import numpy as np
import pandas as pd
import keras
import h5py

#import matplotlib
# matplotlib.use('Agg')
#import matplotlib.pyplot as plt

from keras.utils import to_categorical
# from keras.layers import GRU, LSTM, Dropout, Dense, Input, TimeDistributed, Activation, Flatten, Concatenate
from keras.layers import *
from keras.models import Model, Sequential
from keras.models import load_model
# from keras.callbacks import *
from keras.utils import plot_model


# import tensorflow as tf
# def init():
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = tf.Session(config=config)
#     keras.backend.tensorflow_backend.set_session(session)

# init()


# In[2]:


test_only = 1
model_plot = 0

# path_data = 'data/'
# str_output = 'ans.csv'

if len(sys.argv) == 1 :
    # default setting
    path_data = 'data/'
    str_output = 'ans_cnn.csv'
else :
    path_data = sys.argv[1]
    str_output = sys.argv[2]


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
    
    sample = pd.read_csv('./sample.csv'.format(path_data))
    assert len(sample['phone_sequence']) == len(ans), 'len(sample[\'phone_sequence\']) != len(ans), please check'
    
    sample['phone_sequence'] = pd.DataFrame(ans)
    if test_only :
        sample.to_csv(str_output, index=False)
        
    return sample


# In[9]:


def do_testing_test_only(X_test, lst_size_window, n_CNN_window) :
    #
    # loading data
    #
    #X_test = np.load('./data_pp/X_test_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq))
    model = load_model('./model/{}_{}_{}_{}.h5'.format(model_name, mfcc_or_fbank, n_seq, GL))
    if model_plot :
        plot_model(model, to_file='./model/{}_{}_{}_{}.png'.format(model_name, mfcc_or_fbank, n_seq, GL))
    
    if model_name == 'CNN' :
        X_test = X_test.reshape((-1,n_seq,n_CNN_window,int(dim/n_CNN_window),n_CNN_window))
        print ('X_test.shape : ')
        print (X_test.shape)

    pred = model.predict(X_test)
    for size_window in lst_size_window :
        ans = predict_to_ans(pred, model_name, mfcc_or_fbank, n_seq, GL, size_window, n_CNN_window)
        print (ans[:5])


# In[10]:


lst_size_window = [7] # for pred_to_ans
n_CNN_window = 3
n_seq = 13
# k = 1

mfcc_or_fbank = 'mfcc'
model_name = 'RNN' # CNN or RNN
GL = 'GRU' # GRU or LSTM 

if mfcc_or_fbank == 'mfcc' :
    dim = 39
else :
    dim = 69

# for traingin 
batch_size = 1024

if test_only :
    X_test = preprocessing.preprocessing_test_only(path_data,model_name,mfcc_or_fbank,n_seq,n_CNN_window)
    print ('do testing...')
    do_testing_test_only(X_test, lst_size_window, n_CNN_window)

print ("My program took", str(time.time() - start_time), "to run")
print ('all done')


# In[ ]:



    

