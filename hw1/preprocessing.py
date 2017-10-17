
# coding: utf-8

# In[9]:

# require :
# data/data_pp/lab_train_num.csv because it needs to run 1 hr


# In[10]:

import os
import sys
import pickle
import numpy as np
import pandas as pd


# In[11]:

n_user_train = 462
n_user_test = 74
n_sen_train = 1716
n_sen_test = 342


# In[12]:

if __name__ == '__main__' :
    path_data = 'data/'
    model_name = 'RNN'
    mfcc_or_fbank = 'mfcc'
    n_seq = 15

    if_making_beginEnd = 0

    # RNN seting
    # if_making_RNN_data = 1


# In[13]:

#
# map 48 to char or num
#
def conv_48_to_char_or_num(df_lab_train,path_data,char_or_num='num') :
    path_save = '{}data_pp/'.format(path_data)
    
    map_48_39 = pd.read_csv('{}phones/48_39.map'.format(path_data), header=None, delimiter='\t')
#     print (map_48_39.head(5))
    map_48phone_char = pd.read_csv('{}48phone_char.map'.format(path_data), header=None, delimiter='\t')
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
        
    df_lab_train.to_csv('{}lab_train_num.csv'.format(path_save))

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
            df_beginEnd_train.to_csv('{}data_pp/beginEnd_train.csv'.format(path_data), index=None)

        elif df_id is df_id_test :
            print ('saving beginEnd_test')
            df_beginEnd_test = pd.DataFrame(np.array(lst_beginEnd), columns=['speakerId','sentenceId','index_begin','index_end','length'])
            print (df_beginEnd_test.head(5))
            print (df_beginEnd_test.tail(5))
            df_beginEnd_test.to_csv('{}data_pp/beginEnd_test.csv'.format(path_data), index=None)
            
# if if_making_beginEnd :
#     making_beginEnd()


# In[15]:

#
# making RNN data 
#
# need : beginEnd_train.csv, beginEnd_test.csv

def making_RNN_data(path_data,model_name,mfcc_or_fbank,n_seq) :
    df_y_train = pd.read_csv('{}data_pp/lab_train_num_reindex_axis.csv'.format(path_data))
    df_y_train_noId = df_y_train.drop('0', axis=1)
#     print (df_y_train_noId[:3])
    
    df_train_ark = pd.read_csv('{}{}/train.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
    df_test_ark = pd.read_csv('{}{}/test.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
    df_train_ark_noId = df_train_ark.drop(0, axis=1)
    df_test_ark_noId = df_test_ark.drop(0, axis=1)
#     print (df_train_ark_noId.iloc[:3])
    
    
    df_beginEnd_train = pd.read_csv('{}data_pp/beginEnd_train.csv'.format(path_data))
    df_beginEnd_test = pd.read_csv('{}data_pp/beginEnd_test.csv'.format(path_data))
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
            np.save('{}data_pp/X_train_{}_{}_{}.npy'.format(path_data, model_name, mfcc_or_fbank, n_seq), ary_X_data)
            ary_y_data = np.array(lst_y_data)
            np.save('{}data_pp/y_train_{}_{}_{}.npy'.format(path_data,model_name, mfcc_or_fbank, n_seq), ary_y_data)
        elif df_BE is df_beginEnd_test :
            ary_X_data = np.array(lst_X_data)
            np.save('{}data_pp/X_test_{}_{}_{}.npy'.format(path_data,model_name, mfcc_or_fbank, n_seq), ary_X_data)
    print ('finished making RNN data')
# if if_making_RNN_data :
#     making_RNN_data()


# In[16]:

#
# main (preprocessing)
#
def preprocessing(path_data,model_name,mfcc_or_fbank,n_seq) :
    # just for use. note just use mfcc is enough
    train_ark_no_index_col = pd.read_csv('{}{}/train.ark'.format(path_data,MOF), header=None, delimiter=' ')

    if not os.path.isfile('{}data_pp/lab_train_num.csv'.format(path_data) :
        print ('creating lab_train_num.csv')
        conv_48_to_char_or_num(lab_train,path_data,char_or_num='num')

    if not os.path.isfile('{}data_pp/lab_train_num_reindex_axis.csv'.format(path_data)) :
        print ('creating lab_train_num_reindex_axis.csv')
        lab_train_num = pd.read_csv('{}data_pp/lab_train_num.csv'.format(path_data), index_col=0)
        lab_train_num_reindex_axis = lab_train_num.reindex_axis(train_ark_no_index_col[0], axis=0)
        lab_train_num_reindex_axis.to_csv('{}data_pp/lab_train_num_reindex_axis.csv'.format(path_data))

    if not os.path.isfile('{}data_pp/beginEnd_test.csv'.format(path_data)) :
        print ('creating BE')
        making_beginEnd(path_data,mfcc_or_fbank)

    if not os.path.isfile('{}data_pp/X_test_{}_{}_{}.npy'.format(path_data,model_name, mfcc_or_fbank, n_seq)) :
        print ('creating {}_{}_{}.npy'.format(model_name, mfcc_or_fbank, n_seq))
        making_RNN_data(path_data,model_name,mfcc_or_fbank,n_seq)


    print ('preprocess finished...')
    print ('show the data below : ')

    lab_train_num = pd.read_csv('{}data_pp/lab_train_num.csv'.format(path_data))
    print ('label_train_num.csv : ')
    print (lab_train_num.head(3))

    lab_train_num_reindex_axis = pd.read_csv('{}data_pp/lab_train_num_reindex_axis.csv'.format(path_data))
    print ('lab_train_num_reindex_axis.csv : ')
    print (lab_train_num_reindex_axis.head(5))

    BE_train = pd.read_csv('{}data_pp/beginEnd_train.csv'.format(path_data))
    BE_test = pd.read_csv('{}data_pp/beginEnd_test.csv'.format(path_data))
    print ('beginEnd_train.csv : ')
    print (BE_train.tail(3))
    print ('beginEnd_test.csv : ')
    print (BE_test.tail(3))

    X_test = np.load('{}data_pp/X_test_{}_{}_{}.npy'.format(path_data,model_name, mfcc_or_fbank, n_seq))
    print ('X_test.shape :')
    print (X_test.shape)


# In[ ]:

# with open('{}data_pp/X_train_{}{}.pkl'.format(path_data,mfcc_or_fbank,n_seq), 'rb') as f:
#     lst_X_train = pickle.load(f)
# with open('{}data_pp/y_train_{}{}.pkl'.format(path_data,mfcc_or_fbank,n_seq), 'rb') as f:
#     lst_y_train = pickle.load(f)
# with open('{}data_pp/X_test_{}{}.pkl'.format(path_data,mfcc_or_fbank,n_seq), 'rb') as f:
#     lst_X_test = pickle.load(f)


# In[ ]:




# In[ ]:

# 下面先不要看


# In[ ]:




# In[ ]:

# y_train = pd.read_csv('{}data_pp/lab_train_num_reindex_axis.csv'.format(path_data))
# y_train.drop('0',axis=1, inplace=True)
# y_train.to_csv('{}data_pp/y_train.csv'.format(path_data), index=False)
# y_train = pd.read_csv('{}data_pp/y_train.csv'.format(path_data))
# print (y_train.head(5))


# In[ ]:

# X_train = pd.read_csv('{}{}/train.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
# X_train.drop(0,axis=1, inplace=True)
# X_train.to_csv('{}data_pp/X_train_{}.csv'.format(path_data,mfcc_or_fbank), index=False)
# X_train = pd.read_csv('{}data_pp/X_train_{}.csv'.format(path_data,mfcc_or_fbank))
# print (X_train.head(5))

# y_train = pd.read_csv('{}data_pp/lab_train_num_reindex_axis.csv'.format(path_data))
# y_train.drop('0',axis=1, inplace=True) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! different (0 and'0') because preprocessing above
# y_train.to_csv('{}data_pp/y_train.csv'.format(path_data), index=False)
# y_train = pd.read_csv('{}data_pp/y_train.csv'.format(path_data))
# print (y_train.head(5))

# X_test = pd.read_csv('{}{}/test.ark'.format(path_data,mfcc_or_fbank), header=None, delimiter=' ')
# X_test.drop(0,axis=1, inplace=True)
# X_test.to_csv('{}data_pp/X_test_{}.csv'.format(path_data,mfcc_or_fbank), index=False)
# X_test = pd.read_csv('{}data_pp/X_test_{}.csv'.format(path_data,mfcc_or_fbank))
# print (X_test.head(5))




# In[ ]:

# #
# # preprocessing testing data
# #
# test_ark = pd.read_csv('{}mfcc/test.ark'.format(path_data), header=None, delimiter=' ')
# # print ('test.shape = ' + str(test_ark.shape))
# # print (test_ark.head(5))

# test_ark.drop(0,axis=1, inplace=True)
# test_ark.reset_index(drop=True, inplace=True)
# print (test_ark.head(5))
# print ('test.shape = ' + str(test_ark.shape))


# In[ ]:

# if __name__ == '__main__' :
#     if if_conv_48_to_char_or_num :
#         conv_48_to_char_or_num(lab_train,char_or_num='num')


# In[ ]:



