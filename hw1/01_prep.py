
# coding: utf-8

# In[1]:

import sys
import numpy as np
import pandas as pd


# In[2]:

path_data = 'data/'


# In[ ]:

#
# map 48 to char or num
#
map_48_39 = pd.read_csv('{}phones/48_39.map'.format(path_data), header=None, delimiter='\t')
print (map_48_39.head(5))
map_48phone_char = pd.read_csv('{}48phone_char.map'.format(path_data), header=None, delimiter='\t')
print (map_48phone_char.head(5))
lab_train = pd.read_csv('{}label/train.lab'.format(path_data), index_col=0, header=None)
print (lab_train.head(5))
    
dict_map_39char = dict()
dict_map_39num = dict()
for name in map_48phone_char.iterrows() :
    dict_map_39char[name[1][0]] = name[1][2]
    dict_map_39num[name[1][0]] = name[1][1]
    
dict_map_48char = dict()
dict_map_48num = dict()
for name in map_48_39.iterrows() :
    dict_map_48char[name[1][0]] = dict_map_39char[name[1][1]]
    dict_map_48num[name[1][0]] = dict_map_39num[name[1][1]]

    
print (dict_map_48num)    
len_lab_train = lab_train.shape[0]
for i,lab in enumerate(lab_train[1]) :
    sys.stdout.write('\r{}/{} \t'.format(i,len_lab_train))
    sys.stdout.flush()
    lab_train[1][i] = dict_map_48num[lab]
            
#check
for i,lab in enumerate(lab_train[1]) :
    if lab in dict_map :
        if lab != dict_map[lab] :
            print ('error')
            break

    


# In[ ]:

#
# preprocessing training data (note : sorting between train.ark and train.lab are different)#
# # resorting
# #
# train_ark.sort_values(by=[0], inplace=True)
# lab_train.sort_values(by=[0], inplace=True)
# # print (train_ark.head(5))
# # print (lab_train.head(5))

# train_ark.drop(0,axis=1, inplace=True)
# train_ark.reset_index(drop=True, inplace=True)
# lab_train.drop(0,axis=1, inplace=True)
# lab_train.reset_index(drop=True, inplace=True)
# print (train_ark.head(5))
# print (lab_train.head(5))
# print ('train.shape = ' + str(train_ark.shape))
# print ('lab.shape = ' + str(lab_train.shape))




# In[ ]:

#
# preprocessing testing dat
# In[ ]:




# In[ ]:



