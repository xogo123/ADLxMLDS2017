
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import json


# In[2]:


### path and parameter
path_data = './MLDS_hw2_data/'


# In[3]:


### load data

### load *.npy
lst_name_train_npy = os.listdir('{}training_data/feat'.format(path_data))
lst_name_test_npy = os.listdir('{}testing_data/feat'.format(path_data))

lst_train_npy = []
for npy in lst_name_train_npy :
    lst_train_npy += [np.load('{}training_data/feat/{}'.format(path_data, npy))]
lst_test_npy = []
for npy in lst_name_test_npy :
    lst_test_npy += [np.load('{}testing_data/feat/{}'.format(path_data, npy))]

print ('len of lst_train_npy : '+str(len(lst_train_npy))+'\nshape of train npy : '+str(lst_train_npy[0].shape))
print ('len of lst_test_npy : '+str(len(lst_test_npy))+'\nshape of test npy : '+str(lst_test_npy[0].shape))



# In[ ]:


### load training_label.json testing_label.json
with open('{}training_label.json') as f :
    json_label_train = json.load(f)
    show = json.dump(json_label_train)
    print (show)

