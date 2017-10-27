
# coding: utf-8

# In[1]:


# require :
# data/data_pp/lab_train_num.csv because it needs to run 1 hr

# time(sec) :
# conv_48_to_char_or_num : 2500
# 


# In[2]:


import time
import os
import sys
import pickle
import numpy as np
import pandas as pd

start_time_prep = time.time()


# In[3]:


n_user_train = 462
n_user_test = 74
n_sen_train = 1716
n_sen_test = 342


# In[12]:


if __name__ == '__main__' :
    path_data = 'data/'
    model_name = 'CNN'
    mfcc_or_fbank = 'mfcc'
    n_seq = 13
    n_CNN_window = 3

    if_making_beginEnd = 0

    # RNN seting
    # if_making_RNN_data = 1


# In[13]:


#
# map 48 to char or num
#
def conv_48_to_char_or_num(df_lab_train,path_data,char_or_num='num') :
    if not os.path.isdir('./data_pp') :
        os.mkdir('./data_pp')
    
    map_48_39 = pd.read_csv('./data/phones/48_39.map', header=None, delimiter='\t')
#     print (map_48_39.head(5))
    map_48phone_char = pd.read_csv('./data/48phone_char.map', header=None, delimiter='\t')
#     print (map_48phone_char.head(5))
#     lab_train = pd.read_csv('{}label/train.lab'.format(path_data), index_col=0, header=None)
#     print (lab_train.head(5))

    dict_map_39char = dict()
    dict_map_39num = dict()
    for name in map_48phone_char.iterrows() :
        #dict_map_39char[name[1][0]] = name[1][2]
        dict_map_39num[name[1][0]] = name[1][1]

    dict_map_48char = dict()
    dict_map_48num = dict()
    for name in map_48_39.iterrows() :
        #dict_map_48char[name[1][0]] = dict_map_39char[name[1][1]]
        dict_map_48num[name[1][0]] = dict_map_39num[name[1][1]]


    print (dict_map_48num)    
    len_lab_train = df_lab_train.shape[0]
    for i,lab in enumerate(df_lab_train[1]) :
        sys.stdout.write('\r{}/{} \t'.format(i,len_lab_train))
        sys.stdout.flush()
        df_lab_train[1][i] = dict_map_48num[lab]
        
    df_lab_train.to_csv('./data_pp/lab_train_num.csv')
    print ("conv_48_to_char_or_num took", str(time.time() - start_time_prep), "to run")

    #return df_lab_train


# In[14]:


#
# making beginEnd_train and beginEnd_test
#
def making_beginEnd(path_data,mfcc_or_fbank) :
    df_train_ark = pd.read_csv('{}{}/train.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
    df_test_ark = pd.read_csv('{}{}/test.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')

    df_id_train = df_train_ark[0]
    df_id_test = df_test_ark[0]

    for df_id in [df_id_train, df_id_test] :
        lst_beginEnd = []
        for i,id in enumerate(df_id) :
            lst_id = id.split('_')
            if i == 0:
                index_begin = i
                speakerId = lst_id[0]
                sentenceId = lst_id[1]
            else : 
                frameId = lst_id[2]
                if frameId == '1' :
                    index_end = i - 1
                    length = index_end - index_begin + 1
                    lst_beginEnd += [[speakerId,sentenceId,index_begin,index_end,length]]
                    index_begin = i
                    speakerId = lst_id[0]
                    sentenceId = lst_id[1]
                elif i == len(df_id) - 1 :
                    index_end = i
                    length = index_end - index_begin + 1
                    lst_beginEnd += [[speakerId,sentenceId,index_begin,index_end,length]]
        if df_id is df_id_train :
            print ('saving beginEnd_train')
            df_beginEnd_train = pd.DataFrame(np.array(lst_beginEnd), columns=['speakerId','sentenceId','index_begin','index_end','length'])
            print (df_beginEnd_train.head(5))
            print (df_beginEnd_train.tail(5))
            df_beginEnd_train.to_csv('./data_pp/beginEnd_train.csv', index=None)

        elif df_id is df_id_test :
            print ('saving beginEnd_test')
            df_beginEnd_test = pd.DataFrame(np.array(lst_beginEnd), columns=['speakerId','sentenceId','index_begin','index_end','length'])
            print (df_beginEnd_test.head(5))
            print (df_beginEnd_test.tail(5))
            if not os.path.isdir('./data_pp') :
                os.mkdir('./data_pp')
            df_beginEnd_test.to_csv('./data_pp/beginEnd_test.csv', index=None)
            
# if if_making_beginEnd :
#     making_beginEnd()


# In[15]:


#
# making RNN data 
#
# need : beginEnd_train.csv, beginEnd_test.csv

def making_RNN_data(path_data,model_name,mfcc_or_fbank,n_seq) :
    df_y_train = pd.read_csv('./data_pp/lab_train_num_reindex_axis.csv')
    df_y_train_noId = df_y_train.drop('0', axis=1)
#     print (df_y_train_noId[:3])
    
    df_train_ark = pd.read_csv('{}{}/train.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
    df_test_ark = pd.read_csv('{}{}/test.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
    df_train_ark_noId = df_train_ark.drop(0, axis=1)
    df_test_ark_noId = df_test_ark.drop(0, axis=1)
#     print (df_train_ark_noId.iloc[:3])
    
    
    df_beginEnd_train = pd.read_csv('./data_pp/beginEnd_train.csv')
    df_beginEnd_test = pd.read_csv('./data_pp/beginEnd_test.csv')
#     print (df_beginEnd_train.head(5))
#     print (df_beginEnd_train.tail(5))
#     print (df_beginEnd_test.head(5))
#     print (df_beginEnd_test.tail(5))    
    for df_BE in [df_beginEnd_train,df_beginEnd_test] :
        if df_BE is df_beginEnd_train :
            print ('RNN_train is building...')
            df_ark = df_train_ark_noId
        elif df_BE is df_beginEnd_test :
            print ('RNN_test is building...')
            df_ark = df_test_ark_noId
            
        lst_X_data = []
        lst_y_data = []
        
        for BE in df_BE.iterrows() :
            index_begin = BE[1]['index_begin']
            index_end = BE[1]['index_end']
            length_BE = BE[1]['length']
            n_data = length_BE - n_seq + 1
            assert n_data >= 1, 'n_data should bigger than 1, please do checking'
            
            for i in range(n_data) :
                lst_X_data += [df_ark.iloc[index_begin+i:index_begin+i+n_seq].values.tolist()]
                if df_BE is df_beginEnd_train :
                    lst_y_data += [df_y_train_noId.iloc[index_begin+i:index_begin+i+n_seq].values.tolist()]

        if df_BE is df_beginEnd_train :
            ary_X_data = np.array(lst_X_data)
            np.save('./data_pp/X_train_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq), ary_X_data)
            ary_y_data = np.array(lst_y_data)
            np.save('./data_pp/y_train_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq), ary_y_data)
        elif df_BE is df_beginEnd_test :
            ary_X_data = np.array(lst_X_data)
            np.save('./data_pp/X_test_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq), ary_X_data)
    print ('finished making RNN data')
# if if_making_RNN_data :
#     making_RNN_data()


# In[ ]:


#
# making CNN data 
#
# need : beginEnd_train.csv, beginEnd_test.csv

def making_CNN_data(path_data,model_name,mfcc_or_fbank,n_seq,n_CNN_window) :
    df_y_train = pd.read_csv('./data_pp/lab_train_num_reindex_axis.csv')
    df_y_train_noId = df_y_train.drop('0', axis=1)
#     print (df_y_train_noId[:3])
    
    df_train_ark = pd.read_csv('{}{}/train.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
    df_test_ark = pd.read_csv('{}{}/test.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
    df_train_ark_noId = df_train_ark.drop(0, axis=1)
    df_test_ark_noId = df_test_ark.drop(0, axis=1)
#     print (df_train_ark_noId.iloc[:3])
    
    
    df_beginEnd_train = pd.read_csv('./data_pp/beginEnd_train.csv')
    df_beginEnd_test = pd.read_csv('./data_pp/beginEnd_test.csv')
#     print (df_beginEnd_train.head(5))
#     print (df_beginEnd_train.tail(5))
#     print (df_beginEnd_test.head(5))
#     print (df_beginEnd_test.tail(5))    
    for df_BE in [df_beginEnd_train,df_beginEnd_test] :
        if df_BE is df_beginEnd_train :
            print ('train data is building...')
            df_ark = df_train_ark_noId
        elif df_BE is df_beginEnd_test :
            print ('test data is building...')
            df_ark = df_test_ark_noId
            
        lst_X_data = []
        lst_y_data = []
        
        for BE in df_BE.iterrows() :
            index_begin = BE[1]['index_begin']
            index_end = BE[1]['index_end']
            length_BE = BE[1]['length']
            n_data = length_BE - n_seq + 1 - n_CNN_window + 1
            assert n_data >= 1, 'n_data should bigger than 1, please do checking'
            
            for i in range(n_data) :
                lst_lst_X_data = []
                lst_lst_y_data = []
                for i2 in range(n_seq) :
                    lst_lst_X_data += [df_ark.iloc[index_begin+i+i2:index_begin+i+i2+n_CNN_window].values.tolist()]
#                     if df_BE is df_beginEnd_train :
#                         lst_lst_y_data += [df_y_train_noId.iloc[index_begin+i+i2:index_begin+i+i2+n_CNN_window].values.tolist()]

                lst_X_data += [lst_lst_X_data]
                
                #
                # notice !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #
                
                if df_BE is df_beginEnd_train :
                    lst_y_data += [df_y_train_noId.iloc[index_begin+i+int((n_CNN_window-1)/2):index_begin+i+n_seq+int((n_CNN_window-1)/2)].values.tolist()]

        if df_BE is df_beginEnd_train :
            ary_X_data = np.array(lst_X_data)
            np.save('./data_pp/X_train_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq), ary_X_data)
            ary_y_data = np.array(lst_y_data)
            np.save('./data_pp/y_train_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq), ary_y_data)
        elif df_BE is df_beginEnd_test :
            ary_X_data = np.array(lst_X_data)
            np.save('./data_pp/X_test_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq), ary_X_data)
    print ('finished making CNN data')
# if if_making_RNN_data :
# making_CNN_data(path_data,model_name,mfcc_or_fbank,n_seq)


# In[16]:


#
# main (preprocessing)
#
def preprocessing(path_data,model_name,mfcc_or_fbank,n_seq,n_CNN_window) :
    # just for use. note just use mfcc is enough
    train_ark_no_index_col = pd.read_csv('{}{}/train.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
    
    

    if not os.path.isfile('./data_pp/lab_train_num.csv') :
        print ('creating lab_train_num.csv')
        df_lab_train = pd.read_csv('{}label/train.lab'.format(path_data), index_col=0, header=None)
        conv_48_to_char_or_num(df_lab_train,path_data,char_or_num='num')

    if not os.path.isfile('./data_pp/lab_train_num_reindex_axis.csv') :
        print ('creating lab_train_num_reindex_axis.csv')
        lab_train_num = pd.read_csv('./data_pp/lab_train_num.csv', index_col=0)
        lab_train_num_reindex_axis = lab_train_num.reindex_axis(train_ark_no_index_col[0], axis=0)
        lab_train_num_reindex_axis.to_csv('./data_pp/lab_train_num_reindex_axis.csv')

    if not os.path.isfile('./data_pp/beginEnd_train.csv') :
        print ('creating BE')
        making_beginEnd(path_data,mfcc_or_fbank)

    if not os.path.isfile('./data_pp/X_train_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq)) :
        print ('creating {}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq))
        if model_name == 'RNN' :
            making_RNN_data(path_data,model_name,mfcc_or_fbank,n_seq)
        elif model_name == 'CNN' :
            making_CNN_data(path_data,model_name,mfcc_or_fbank,n_seq,n_CNN_window)


    print ('preprocess finished...')
#     print ('show the data below : ')

#     lab_train_num = pd.read_csv('./data_pp/lab_train_num.csv')
#     print ('label_train_num.csv : ')
#     print (lab_train_num.head(3))

#     lab_train_num_reindex_axis = pd.read_csv('./data_pp/lab_train_num_reindex_axis.csv')
#     print ('lab_train_num_reindex_axis.csv : ')
#     print (lab_train_num_reindex_axis.head(5))

#     BE_train = pd.read_csv('./data_pp/beginEnd_train.csv')
#     BE_test = pd.read_csv('./data_pp/beginEnd_test.csv')
#     print ('beginEnd_train.csv : ')
#     print (BE_train.tail(3))
#     print ('beginEnd_test.csv : ')
#     print (BE_test.tail(3))

#     X_test = np.load('./data_pp/X_test_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq))
#     print ('X_test.shape :')
#     print (X_test.shape)


# In[ ]:


# 下面是test_only


# In[ ]:


#
# making beginEnd_train and beginEnd_test
#
def making_beginEnd_test_only(path_data,mfcc_or_fbank) :
    start_time_tmp = time.time()
#     df_train_ark = pd.read_csv('{}{}/train.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
    df_test_ark = pd.read_csv('{}{}/test.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')

#     df_id_train = df_train_ark[0]
    df_id_test = df_test_ark[0]

    for df_id in [df_id_test] :
        lst_beginEnd = []
        for i,id in enumerate(df_id) :
            lst_id = id.split('_')
            if i == 0:
                index_begin = i
                speakerId = lst_id[0]
                sentenceId = lst_id[1]
            else : 
                frameId = lst_id[2]
                if frameId == '1' :
                    index_end = i - 1
                    length = index_end - index_begin + 1
                    lst_beginEnd += [[speakerId,sentenceId,index_begin,index_end,length]]
                    index_begin = i
                    speakerId = lst_id[0]
                    sentenceId = lst_id[1]
                elif i == len(df_id) - 1 :
                    index_end = i
                    length = index_end - index_begin + 1
                    lst_beginEnd += [[speakerId,sentenceId,index_begin,index_end,length]]
#         if df_id is df_id_train :
#             print ('saving beginEnd_train')
#             df_beginEnd_train = pd.DataFrame(np.array(lst_beginEnd), columns=['speakerId','sentenceId','index_begin','index_end','length'])
#             print (df_beginEnd_train.head(5))
#             print (df_beginEnd_train.tail(5))
#             df_beginEnd_train.to_csv('./data_pp/beginEnd_train.csv', index=None)

#         elif df_id is df_id_test :
        print ('saving beginEnd_test')
        df_beginEnd_test = pd.DataFrame(np.array(lst_beginEnd), columns=['speakerId','sentenceId','index_begin','index_end','length'])
        print (df_beginEnd_test.head(5))
        print (df_beginEnd_test.tail(5))
        if not os.path.isdir('./data_pp') :
            os.mkdir('./data_pp')
        df_beginEnd_test.to_csv('./data_pp/beginEnd_test.csv', index=None)
    print ("making_beginEnd_test_only took", str(time.time() - start_time_tmp), "to run")

# if if_making_beginEnd :
#     making_beginEnd()


# In[ ]:


#
# making RNN data 
#
# need : beginEnd_train.csv, beginEnd_test.csv

def making_RNN_data_test_only(path_data,model_name,mfcc_or_fbank,n_seq) :
    start_time_tmp = time.time()
#     df_y_train = pd.read_csv('./data_pp/lab_train_num_reindex_axis.csv')
#     df_y_train_noId = df_y_train.drop('0', axis=1)
#     print (df_y_train_noId[:3])
    
#     df_train_ark = pd.read_csv('{}{}/train.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
    df_test_ark = pd.read_csv('{}{}/test.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
#     df_train_ark_noId = df_train_ark.drop(0, axis=1)
    df_test_ark_noId = df_test_ark.drop(0, axis=1)
#     print (df_train_ark_noId.iloc[:3])
    
    
#     df_beginEnd_train = pd.read_csv('./data_pp/beginEnd_train.csv')
    df_beginEnd_test = pd.read_csv('./data_pp/beginEnd_test.csv')
#     print (df_beginEnd_train.head(5))
#     print (df_beginEnd_train.tail(5))
#     print (df_beginEnd_test.head(5))
#     print (df_beginEnd_test.tail(5))    
    for df_BE in [df_beginEnd_test] :
        if df_BE is df_beginEnd_test :
            print ('RNN_test is building...')
            df_ark = df_test_ark_noId
            
        lst_X_data = []
        lst_y_data = []
        
        for BE in df_BE.iterrows() :
            index_begin = BE[1]['index_begin']
            index_end = BE[1]['index_end']
            length_BE = BE[1]['length']
            n_data = length_BE - n_seq + 1
            assert n_data >= 1, 'n_data should bigger than 1, please do checking'
            
            for i in range(n_data) :
                lst_X_data += [df_ark.iloc[index_begin+i:index_begin+i+n_seq].values.tolist()]
#         if df_BE is df_beginEnd_train :
#             ary_X_data = np.array(lst_X_data)
#             np.save('./data_pp/X_train_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq), ary_X_data)
#             ary_y_data = np.array(lst_y_data)
#             np.save('./data_pp/y_train_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq), ary_y_data)
#         elif df_BE is df_beginEnd_test :
        ary_X_data = np.array(lst_X_data)
        np.save('./data_pp/X_test_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq), ary_X_data)
    print ('finished making RNN data')
    print ("making_RNN_data_test_only took", str(time.time() - start_time_tmp), "to run")
    return ary_X_data
# if if_making_RNN_data :
#     making_RNN_data()


# In[ ]:


#
# making CNN data 
#
# need : beginEnd_train.csv, beginEnd_test.csv

def making_CNN_data_test_only(path_data,model_name,mfcc_or_fbank,n_seq,n_CNN_window) :
    start_time_tmp = time.time()
#     df_y_train = pd.read_csv('./data_pp/lab_train_num_reindex_axis.csv')
#     df_y_train_noId = df_y_train.drop('0', axis=1)
#     print (df_y_train_noId[:3])
    
#     df_train_ark = pd.read_csv('{}{}/train.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
    df_test_ark = pd.read_csv('{}{}/test.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
#     df_train_ark_noId = df_train_ark.drop(0, axis=1)
    df_test_ark_noId = df_test_ark.drop(0, axis=1)
#     print (df_train_ark_noId.iloc[:3])
    
    
#     df_beginEnd_train = pd.read_csv('./data_pp/beginEnd_train.csv')
    df_beginEnd_test = pd.read_csv('./data_pp/beginEnd_test.csv')
    
    for df_BE in [df_beginEnd_test] :
        print ('test data is building...')
        df_ark = df_test_ark_noId
            
        lst_X_data = []
        lst_y_data = []
        
        for BE in df_BE.iterrows() :
            index_begin = BE[1]['index_begin']
            index_end = BE[1]['index_end']
            length_BE = BE[1]['length']
            n_data = length_BE - n_seq + 1 - n_CNN_window + 1
            assert n_data >= 1, 'n_data should bigger than 1, please do checking'
            
            for i in range(n_data) :
                lst_lst_X_data = []
                lst_lst_y_data = []
                for i2 in range(n_seq) :
                    lst_lst_X_data += [df_ark.iloc[index_begin+i+i2:index_begin+i+i2+n_CNN_window].values.tolist()]
#                     if df_BE is df_beginEnd_train :
#                         lst_lst_y_data += [df_y_train_noId.iloc[index_begin+i+i2:index_begin+i+i2+n_CNN_window].values.tolist()]

                lst_X_data += [lst_lst_X_data]
                
                #
                # notice !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                #
                
#                 if df_BE is df_beginEnd_train :
#                     lst_y_data += [df_y_train_noId.iloc[index_begin+i+int((n_CNN_window-1)/2):index_begin+i+n_seq+int((n_CNN_window-1)/2)].values.tolist()]

#         if df_BE is df_beginEnd_train :
#             ary_X_data = np.array(lst_X_data)
#             np.save('./data_pp/X_train_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq), ary_X_data)
#             ary_y_data = np.array(lst_y_data)
#             np.save('./data_pp/y_train_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq), ary_y_data)
#         elif df_BE is df_beginEnd_test :
        ary_X_data = np.array(lst_X_data)
        print (str(time.time() - start_time_tmp))
        np.save('./data_pp/X_test_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq), ary_X_data)
        print (str(time.time() - start_time_tmp))
    print ('finished making CNN data')
    print ("making_CNN_data_test_only took", str(time.time() - start_time_tmp), "to run")
    return ary_X_data
# if if_making_RNN_data :
# making_CNN_data(path_data,model_name,mfcc_or_fbank,n_seq)


# In[ ]:


#
# main (preprocessing)
#
def preprocessing_test_only(path_data,model_name,mfcc_or_fbank,n_seq,n_CNN_window) :
#     # just for use. note just use mfcc is enough
#     train_ark_no_index_col = pd.read_csv('{}{}/train.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
    
    

#     if not os.path.isfile('./data_pp/lab_train_num.csv') :
#         print ('creating lab_train_num.csv')
#         df_lab_train = pd.read_csv('{}label/train.lab'.format(path_data), index_col=0, header=None)
#         conv_48_to_char_or_num(df_lab_train,path_data,char_or_num='num')

#     if not os.path.isfile('./data_pp/lab_train_num_reindex_axis.csv') :
#         print ('creating lab_train_num_reindex_axis.csv')
#         lab_train_num = pd.read_csv('./data_pp/lab_train_num.csv', index_col=0)
#         lab_train_num_reindex_axis = lab_train_num.reindex_axis(train_ark_no_index_col[0], axis=0)
#         lab_train_num_reindex_axis.to_csv('./data_pp/lab_train_num_reindex_axis.csv')

    if not os.path.isfile('./data_pp/beginEnd_test.csv'.format(path_data)) :
        print ('creating BE')
        making_beginEnd_test_only(path_data,mfcc_or_fbank)

    if not os.path.isfile('./data_pp/X_test_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq)) :
        print ('creating {}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq))
        if model_name == 'RNN' :
            ary_X_data = making_RNN_data_test_only(path_data,model_name,mfcc_or_fbank,n_seq)
        elif model_name == 'CNN' :
            ary_X_data = making_CNN_data_test_only(path_data,model_name,mfcc_or_fbank,n_seq,n_CNN_window)
    else :
        print ('preprocess finished...')
        return 0


    print ('preprocess finished...')
    return ary_X_data
#     print ('show the data below : ')

#     lab_train_num = pd.read_csv('./data_pp/lab_train_num.csv')
#     print ('label_train_num.csv : ')
#     print (lab_train_num.head(3))

#     lab_train_num_reindex_axis = pd.read_csv('./data_pp/lab_train_num_reindex_axis.csv')
#     print ('lab_train_num_reindex_axis.csv : ')
#     print (lab_train_num_reindex_axis.head(5))

#     BE_train = pd.read_csv('./data_pp/beginEnd_train.csv')
#     BE_test = pd.read_csv('./data_pp/beginEnd_test.csv')
#     print ('beginEnd_train.csv : ')
#     print (BE_train.tail(3))
#     print ('beginEnd_test.csv : ')
#     print (BE_test.tail(3))

#     X_test = np.load('./data_pp/X_test_{}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq))
#     print ('X_test.shape :')
#     print (X_test.shape)

