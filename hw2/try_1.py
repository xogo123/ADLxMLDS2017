
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import json
import torch

import tensorflow as tf
import keras
import matplotlib
import matplotlib.pyplot as plt 
def init():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)
init()


# In[2]:


### path and parameter
path_data = './MLDS_hw2_data/'
model_name = 's2s'


# In[ ]:


class Lang:
    def __init__(self):
        self.word2index = {"<BOS>": 0, "<EOS>" :1}
        self.word2count = {"<BOS>": 0, "<EOS>" : 0}
        self.index2word = {0: "<BOS>", 1: "<EOS>"}
        self.n_words = 2  # Count BOS and EOS
        self.max_len_seq = 0

    def addSentence(self, sentence):
        lst_word = sentence.split()
        if self.max_len_seq < len(lst_word) :
            self.max_len_seq = len(lst_word)
        for word in lst_word :
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# In[3]:


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


# In[4]:


### load data

### load *.npy
# lst_name_train_npy = os.listdir('{}training_data/feat'.format(path_data))
# lst_name_test_npy = os.listdir('{}testing_data/feat'.format(path_data))

# lst_train_npy = []
# for npy in lst_name_train_npy :
#     lst_train_npy += [np.load('{}training_data/feat/{}'.format(path_data, npy))]
# lst_test_npy = []
# for npy in lst_name_test_npy :
#     lst_test_npy += [np.load('{}testing_data/feat/{}'.format(path_data, npy))]

# print ('len of lst_train_npy : '+str(len(lst_train_npy))+'\nshape of train npy : '+str(lst_train_npy[0].shape))
# print ('len of lst_test_npy : '+str(len(lst_test_npy))+'\nshape of test npy : '+str(lst_test_npy[0].shape))

### load training_label.json testing_label.json
with open('{}training_label.json'.format(path_data)) as f :
    lst_dict_label_train = json.load(f)
print ('\ntraining_label.json : ')
print ('caption : \n' + str(lst_dict_label_train[0]['caption']))
print ('id : \n' + str(lst_dict_label_train[0]['id']))


# In[5]:


class Lang:
    def __init__(self):
        self.word2index = {"<BOS>": 0, "<EOS>" :1}
        self.word2count = {"<BOS>": 0, "<EOS>" : 0}
        self.index2word = {0: "<BOS>", 1: "<EOS>"}
        self.n_words = 2  # Count BOS and EOS
        self.max_len_seq = 0

    def addSentence(self, sentence):
        lst_word = sentence.split()
        if self.max_len_seq < len(lst_word) :
            self.max_len_seq = len(lst_word)
        for word in lst_word :
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# In[6]:


lang = Lang()
for dict_label_train in lst_dict_label_train :
    for sentence in dict_label_train['caption'] :
        sentence = sentence[:-1] + ' <EOS>'
        lang.addSentence(sentence) # remove "."
print (lang.n_words)
print (lang.max_len_seq)
# assert lang.word2count['<BOS>'] == lang.word2count['<EOS>'], number of "<BOS>" != number of "<EOS>"


# In[7]:


### index to one-hot
def Str2OneHot(sentence, n_class, dict_map, max_len_seq) :
    ### sentence to lst_index_sentence
    sentence = sentence[:-1] + ' <EOS>'
    lst_word = sentence.split()
    ary_oneHot = np.zeros((max_len_seq,n_class))
    lst_index_word = [dict_map[word] for word in lst_word]
    ary_oneHot[range(len(lst_word)),lst_index_word] = 1
    ary_oneHot[range(len(lst_word),len(ary_oneHot)),:] = 0.01 # others all set <EOS>
    return ary_oneHot
    
### just for test
ary = Str2OneHot('A woman goes under a horse.', lang.n_words, lang.word2index, lang.max_len_seq)
print (ary.shape)
        


# In[8]:


for i in range(3,5) :
    print (i)


# In[9]:


lst_id = [dict_label_train['id'] for dict_label_train in lst_dict_label_train]
lst_train_EC_input = []
lst_train_DC_output = []
for i, id in enumerate(lst_id) :
    npy = np.load('{}training_data/feat/{}.npy'.format(path_data, id))
    for caption in lst_dict_label_train[i]['caption'][:3] :
        lst_train_EC_input += [npy]
        ary_OneHot = Str2OneHot(caption, lang.n_words, lang.word2index, lang.max_len_seq)
        lst_train_DC_output += [ary_OneHot]
assert len(lst_train_EC_input) == len(lst_train_DC_output), "??"
ary_train_EC_input = np.concatenate(lst_train_EC_input,axis=0).reshape(-1,80,4096)
ary_train_DC_output = np.concatenate(lst_train_DC_output,axis=0).reshape(-1,lang.max_len_seq,lang.n_words)
# ary_train_EC_input = np.vstack(tuple(lst_train_EC_input)).reshape(-1,80,4096)
print (ary_train_EC_input.shape)


# In[10]:


import keras
from keras.layers import *
from keras.models import *
from keras import backend as K


# In[11]:


def model_pretrain(lang=lang) :

    EC_input = Input(shape=(80,4096))
    EC_output, EC_output_state = GRU(32,return_state=True)(EC_input)
    print (EC_output_state)
    DC_input = Input(shape=(None,lang.n_words))
    DC_input_M = Masking(mask_value=0.01)(DC_input)
    print (DC_input)

    DC_gru = GRU(32, return_sequences=True)
    DC_time_dense = TimeDistributed(Dense(lang.n_words, activation='softmax'))

#     DC_output_state = EC_output_state
#     DC_output = DC_input
#     lst_DC_output = []
#     for _ in range(lang.max_len_seq) :
    DC_output = DC_gru(DC_input_M, initial_state=EC_output_state)
    DC_output = DC_time_dense(DC_output)

    model = Model([EC_input,DC_input],DC_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print (model.summary())
    return model


# In[12]:




# EC_input = Input(shape=(80,4096))
# EC_output, EC_output_state = GRU(32,return_state=True)(EC_input)
# print (EC_output_state)
# DC_input = Input(shape=(1,lang.n_words))
# print (DC_input)

# DC_gru = GRU(32, return_sequences=True, return_state=True)
# DC_dense = Dense(lang.n_words, activation='softmax')

# DC_output_state = EC_output_state
# DC_output = DC_input
# lst_DC_output = []
# for _ in range(lang.max_len_seq) :
#     DC_output, DC_output_state = DC_gru(DC_output, initial_state=DC_output_state)
#     DC_output = DC_dense(DC_output)
#     lst_DC_output += [DC_output]

# DC_output = Lambda(lambda x: K.concatenate(x, axis=1))(lst_DC_output)

    
# model = Model([EC_input,DC_input],DC_output)
# model.compile(optimizer='adam', loss='categorical_crossentropy')
# print (model.summary())


# In[ ]:


ary_temp = np.zeros((len(ary_train_DC_output),1,lang.n_words))
ary_temp[:,0,0] = 1
print (ary_temp.shape)
print (ary_train_DC_output[:,:-1].shape)

ary_train_DC_input = np.concatenate([ary_temp,ary_train_DC_output[:,:-1]],axis=1)
# for i in range(len(ary_train_DC_output)) :
#     print (i)
#     if i == 0 :
#         lst_train_DC_input = [np.concatenate([ary_temp,ary_train_DC_output[i,:-1]],axis=1).tolist()]
#     else :
#         lst_train_DC_input += [np.concatenate([ary_temp,ary_train_DC_output[i,:-1]],axis=1).tolist()]
# ary_train_DC_output = np.asarray(lst_train_DC_input)
print (ary_train_DC_input[:7])

if not os.path.isdir('./model') :
    os.mkdir('./model')
k = 0
while 1 :
    if os.path.isfile('./model/{}_{}.h5'.format(model_name,k)) :
        k += 1
    else :
        break
MCP = keras.callbacks.ModelCheckpoint('./model/{}_{}.h5'.format(model_name,k), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
ES = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=0, mode='auto')

model = model_pretrain(lang)

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
loading_model = 0

if loading_model :
    model = load_model('./model/s2s_0.h5')
log = model.fit([ary_train_EC_input, ary_train_DC_input],ary_train_DC_output, epochs=500, batch_size=32, validation_split=0.1, callbacks=[ES]) 
df_log = log.history
fig = keras_log_plot(df_log)



# In[ ]:


def ary_pred_to_sentence(ary_pred) :
    ary_pred_argmax = np.argmax(ary_pred,axis=2)
    lst_ans_numbers = ary_pred_argmax.tolist()
#     print (lst_ans_numbers[:3])
    lst_ans_string = []
    for ans_numbers in lst_ans_numbers :
        lst_ans_string += [' '.join([lang.index2word[ans] for ans in ans_numbers])]
    return lst_ans_string


# In[ ]:


ary_pred = model.predict([ary_train_EC_input, ary_train_DC_input])


# In[ ]:


# print (ary_train_EC_input[:3])
# print (ary_train_DC_input[1]
print (ary_train_DC_input.shape)
print (np.argmax(ary_train_DC_input[1],axis=1))
print (np.argmax(ary_train_DC_input[1],axis=1).shape)
print (ary_pred.shape)
ary_pred_argmax = np.argmax(ary_pred,axis=2)
print (ary_pred_argmax.shape)
# print (ary_pred_argmax[:5])

lst_ans_string = ary_pred_to_sentence(ary_pred)
print (lst_ans_string[:5])
# print (len(lst_ans_string[0]))


# In[ ]:


s = []
s+=[['aa']]
s+=['bb']
print (s)
print (len(lst_ans_string[2]))


# In[ ]:


# if not os.path.isdir('./model') :
#     os.mkdir('./model')
# model.save('./model/s2s_2.h5')


# In[ ]:


max_len_seq = 30

ary_train_EC_input = np.concatenate(lst_train_npy,axis=0)
ary_train_EC_input = ary_train.reshape(-1,80,4096)
print (ary_train_EC_input.shape)

ary_train_DC_output = 


# In[ ]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()
if use_cuda :
    print ('using cuda')


# In[ ]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size=4096, hidden_size=32, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=self.n_layers, batch_first=True, bidirectional=False)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1,self.n_layers, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


# In[ ]:


class DecoderRNN(nn.Module) :
    def __init__(self, hidden_size=32, input_size=6871, n_layers=1) :
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=self.n_layers, batch_first=True, bidirectional=False)
        self.out = nn.Linear(hidden_size, input_size)
        self.softmax = nn.LogSoftmax()
        
    def forward(self, input, hidden) :
        output = input
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        output = self.out(output.float())
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        output = self.softmax(output)
        return output, hidden


# In[ ]:


# rnn = nn.GRU(10, 20, 2)
# rnn = rnn.cuda()
# input = Variable(torch.randn(5, 3, 10)).cuda()
# h0 = Variable(torch.randn(2, 3, 20)).cuda()
# output, hn = rnn(input, h0)


# In[ ]:


E_net = EncoderRNN(4096,64,1).cuda()
D_net = DecoderRNN(64,6871,1).cuda()
v_input = Variable(torch.from_numpy(lst_train_npy[0].reshape((1,80,4096))))
v_input = v_input.cuda()
h_0 = E_net.initHidden()
output_encoder, hidden = E_net(v_input, h_0)

lst_output = []
print (lst_dict_label_train[0]['caption'])
for caption in lst_dict_label_train[0]['caption'] :
    ary_label_OneHot = Str2OneHot(caption, lang.n_words, lang.word2index)
    print (ary_label_OneHot.shape)
    i = 0
    for i in range(len(ary_label_OneHot)) : # check not <EOS>
        print (i)
        output = Variable(torch.from_numpy(ary_label_OneHot[i].reshape(1,1,-1))).cuda()
        output, hidden = D_net.forward(output, hidden)
        lst_output += [output]
        i += 1
                

target = Variable(torch.from_numpy(ary_label_OneHot[1:])).cuda()  # a dummy target, for example
criterion = nn.CrossEntropyLoss().cuda()
print (lst_output[0])
# print (target[0])
# print (lst_output[0].data.shape)
# a = Variable(torch.zeros())
print (target[0].data.shape)

# loss = criterion()
loss = criterion(lst_output[0].view(1,6871), target[0])
print (loss)
for i in range(len(label_OneHot)) :
    loss.add
loss.backward()
optimizer.step()
print ('epoch : {}'.format(epoch) + '\t loss : ' + str(loss.data[0]))
# print ('epoch : {}'.format(epoch) + '\t loss : ' + str(loss.data[0]))



# for epoch in range(10) :
#     running_loss = 0.0
#     loss = criterion(lst_output[0], target[i]) for i in range(len(label_OneHot)))
#     loss.backward()
#     optimizer.step()
#     print ('epoch : {}'.format(epoch) + '\t loss : ' + str(loss.data[0]))








# In[ ]:


### note
# remember to add <BOS> and <EOS>
# Tom's should split as "TOM" and "s"
# notice [:-1] to delete "."
# A and a


# In[ ]:


# ### build dict_map_W2I and dict_map_I2W to mapping from word to index

# ### count unique words
# lst_word = list(set([word for dict_label_train in lst_dict_label_train for str_label in dict_label_train['caption'] for word in str_label[:-1].split()]))
# lst_word += ['<BOS>']
# lst_word += ['<EOS>']
# n_class = len(lst_word)
# print ('len(lst_word) : {}'.format(n_class))

# ### map from word to index
# dict_map_W2I = dict()
# index = 0
# for word in lst_word :
#     if word not in dict_map_W2I :
#         dict_map_W2I[word] = index
#         index += 1
        
# ### map from index to word
# dict_map_I2W = {v:k for k,v in dict_map_W2I.items()}

# print ('<<-- dict_map_W2I and dict_I2W are built completely -->>')


# In[ ]:


a = np.arange(15).reshape((3,5))
max_size=5
for i in range(max_size-len(a)) :
    print (i)
    a = np.append(a,[[0,0,0,0,0]],axis=0)
# a = np.pad(a,(0,10-len(a)),'constant')
print (a)


# In[ ]:


a = sum(i for i in range(5))
print (a)

