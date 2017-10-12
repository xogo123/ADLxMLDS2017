
# coding: utf-8
# preprocessing
# auther : Felix Chu
# 2017_10_06

# note : many different people will say same sentences

# train.lab
# 48_39.map
# 48phone_char.map
# sample.csv
# mfcc/train.ark
#     mfcc/test.ark
#     fbank/train.ark
#     fbank/test.ark

# check_same_user
# check_same_sentence
# In[1]:

import numpy as np
import pandas as pd


# In[2]:

#
# train.lab
#
path_data = 'data/'
lab_train = pd.read_csv('{}label/train.lab'.format(path_data), header=None)
#print (lab_train.head(5))
print (lab_train[9960:9970])
print (lab_train.shape)



# In[3]:

#
# 48_39.map
#
map_48_39 = pd.read_csv('{}phones/48_39.map'.format(path_data), header=None, delimiter='\t')
print ('\n48_39.map')
print (map_48_39.head(5))



# In[6]:

#
# 48phone_char.map
#
map_48phone_char = pd.read_csv('{}48phone_char.map'.format(path_data), header=None, delimiter='\t')
print ('\n48phone_char.map')
print (map_48phone_char)



# In[7]:

#
# sample.csv
#
sample = pd.read_csv('{}sample.csv'.format(path_data))
print (sample.head(5))


# In[6]:

#
# /mfcc/train.ark
#
train_ark = pd.read_csv('{}mfcc/train.ark'.format(path_data), header=None, delimiter=' ')
print (train_ark.head(5))


# In[7]:

#
# check if same user in both train set and test set
#

train_ark = pd.read_csv('{}mfcc/train.ark'.format(path_data), header=None, delimiter=' ')
test_ark = pd.read_csv('{}mfcc/test.ark'.format(path_data), header=None, delimiter=' ')

id_train = train_ark[0]
id_test = test_ark[0]

lst_user_train = []
lst_user_test = []

for id in id_train :
    name = id.split('_')[0]
    if name not in lst_user_train :
        lst_user_train += [name]
for id in id_test :
    name = id.split('_')[0]
    if name not in lst_user_test :
        lst_user_test += [name]

print ('train users : ' + str(len(lst_user_train)))
print ('test users : ' + str(len(lst_user_test)))

intersection = set(lst_user_train) & set(lst_user_test)
if intersection :
    print ('they have same user')
else :
    print ('they don\'t have same user')


# In[8]:

#
# check if same sentence in both train set and test set
#
train_ark = pd.read_csv('{}mfcc/train.ark'.format(path_data), header=None, delimiter=' ')
test_ark = pd.read_csv('{}mfcc/test.ark'.format(path_data), header=None, delimiter=' ')

id_train = train_ark[0]
id_test = test_ark[0]

lst_sen_train = []
lst_sen_test = []

for id in id_train :
    name = id.split('_')[1]
    if name not in lst_sen_train :
        lst_sen_train += [name]
for id in id_test :
    name = id.split('_')[1]
    if name not in lst_sen_test :
        lst_sen_test += [name]

print ('train sentences : ' + str(len(lst_sen_train)))
print ('test sentences : ' + str(len(lst_sen_test)))

intersection = set(lst_sen_train) & set(lst_sen_test)
if intersection :
    print ('they have same sentence')
else :
    print ('they don\'t have same sentence')


# In[ ]:




# In[ ]:



