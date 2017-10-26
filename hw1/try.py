
# coding: utf-8

# In[3]:

import keras
from keras.models import load_model

model_name = 'CNN'
mfcc_or_fbank = 'mfcc'
# n_seq = 

# model = load_model('./model/{}_{}_{}_{}_{}.h5'.format(model_name, mfcc_or_fbank, n_seq, GL, k))
model = load_model('./model/best.h5')
print (model.summary())


# In[41]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
log = pd.DataFrame(np.arange(15)[::-1].reshape((3,5)))
print (a[[0,3]])

matplotlib.rcParams.update({'font.size': 16})
fig = plt.figure(1,figsize=(20,10))
plt.subplot(121)
plt.plot(log[0], label='train_acc')
plt.plot(log[1], label='val_acc')
plt.legend(fontsize=20)
plt.xlabel('epoch', fontsize=20, color='black')
plt.ylabel('acc', fontsize=20, color='black')

plt.subplot(122)
plt.plot(log[2], label='train_loss')
plt.plot(log[3], label='val_loss')
plt.legend(fontsize=20)
plt.xlabel('epoch', fontsize=20, color='black')
plt.ylabel('loss', fontsize=20, color='black')
plt.show()
fig.savefig('./111.png')


# In[ ]:

def keras_log_plot(log) :
    matplotlib.rcParams.update({'font.size': 16})
    fig = plt.figure(1,figsize=(20,10))
    
    plt.subplot(121)
    plt.plot(log['acc'], label='train_acc')
    plt.plot(log['val_acc'], label='val_acc')
    plt.legend(fontsize=20)
    plt.xlabel('epoch', fontsize=20, color='black')
    plt.ylabel('acc', fontsize=20, color='black')

    plt.subplot(122)
    plt.plot(log['loss'], label='train_loss')
    plt.plot(log['val_loss'], label='val_loss')
    plt.legend(fontsize=20)
    plt.xlabel('epoch', fontsize=20, color='black')
    plt.ylabel('loss', fontsize=20, color='black')
    return fig

