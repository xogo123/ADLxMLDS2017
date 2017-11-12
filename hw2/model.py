
# coding: utf-8

# In[1]:


import os
import sys
import numpy as np
import json
import torch
import pickle

import tensorflow as tf
import keras
from keras.layers import *
from keras.models import *
from keras import backend as K

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
max_seq = 8
n_caption = -1

loading_model = 1
do_training = 1
teacherForce = 0

save_model = 1
train_data_loading = 1
test_data_loading = 1
peer_review_data_loading = 0

special_task = 1


# In[3]:


class Lang:
    def __init__(self):
        self.word2index = {"<BOS>": 0, "<EOS>" :1}
        self.word2count = {"<BOS>": 0, "<EOS>" : 0}
        self.index2word = {0: "<BOS>", 1: "<EOS>"}
        self.n_words = 2  # Count BOS and EOS
        self.max_len_seq = 0

    def addSentence(self, sentence):
        lst_word = sentence.split()
        if len(lst_word) > max_seq :
            return 0
        elif self.max_len_seq < len(lst_word) :
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


# In[4]:


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


# In[5]:


# ### load data

# ### load *.npy
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


# In[6]:


# def loading_npy(path_data, dir_name, id_txt_name) :
#     ### name in ['training_data','testing_data','peer_review']
#     ### load data

#     ### load *.npy
#     with open('{}{}.txt'.format(path_data, id_txt_name)) as f :
#         lst_id = f.readlines()
        
    
    
#     lst_name_train_npy = os.listdir('{}training_data/feat'.format(path_data))
#     lst_name_test_npy = os.listdir('{}testing_data/feat'.format(path_data))

#     lst_train_npy = []
#     for npy in lst_name_train_npy :
#         lst_train_npy += [np.load('{}training_data/feat/{}'.format(path_data, npy))]
#     lst_test_npy = []
#     for npy in lst_name_test_npy :
#         lst_test_npy += [np.load('{}testing_data/feat/{}'.format(path_data, npy))]

#     print ('len of lst_train_npy : '+str(len(lst_train_npy))+'\nshape of train npy : '+str(lst_train_npy[0].shape))
#     print ('len of lst_test_npy : '+str(len(lst_test_npy))+'\nshape of test npy : '+str(lst_test_npy[0].shape))

#     return lst_name, 
# ### load training_label.json testing_label.json


# In[7]:


def lang_init(lst_dict_label_train) :
    lang = Lang()
    for dict_label_train in lst_dict_label_train :
        for sentence in dict_label_train['caption'] :
            sentence = sentence[:-1] + ' <EOS>'
            lang.addSentence(sentence) # remove "."
    print (lang.n_words)
    print (lang.max_len_seq)
    assert lang.max_len_seq == max_seq, 'error here'
    return lang
# assert lang.word2count['<BOS>'] == lang.word2count['<EOS>'], number of "<BOS>" != number of "<EOS>"


# In[8]:


### index to one-hot
def Str2OneHot(sentence, n_class, dict_map, max_len_seq) :
    ### sentence to lst_index_sentence
    sentence = sentence[:-1] + ' <EOS>'
    lst_word = sentence.split()
    ary_oneHot = np.zeros((max_len_seq,n_class))
    lst_index_word = [dict_map[word] for word in lst_word]
    ary_oneHot[range(len(lst_word)),lst_index_word] = 1
    ary_oneHot[range(len(lst_word),len(ary_oneHot)),:] = 0.0 # others all set <EOS>
    return ary_oneHot
    
### just for test
# ary = Str2OneHot('A woman goes under a horse.', lang.n_words, lang.word2index, lang.max_len_seq)
# print (ary.shape)
        


# In[9]:


### preprocessing

### for training data
### load lst_dict_label_train

with open('{}training_label.json'.format(path_data)) as f :
    lst_dict_label_train = json.load(f)
print ('\ntraining_label.json : ')
print ('caption : \n' + str(lst_dict_label_train[0]['caption'][:2]))
print ('id : \n' + str(lst_dict_label_train[0]['id']))

lang = lang_init(lst_dict_label_train)

if train_data_loading :
    ### pair caption and vedio feature
    ### output : ary_train_EC_input
    ###          ary_train_DC_output
    ###          ary_train_DC_input
    lst_id_train = [dict_label_train['id'] for dict_label_train in lst_dict_label_train]
    lst_train_EC_input = []
    lst_train_DC_output = []
    for i, id in enumerate(lst_id_train) :
        lst_npy = np.load('{}training_data/feat/{}.npy'.format(path_data, id)).tolist()
        for caption in lst_dict_label_train[i]['caption'][:n_caption] :
            if len(caption.split()) >= max_seq : # note >=
                continue
            lst_train_EC_input += [lst_npy]
            lst_ary_OneHot = Str2OneHot(caption, lang.n_words, lang.word2index, lang.max_len_seq).tolist()
            lst_train_DC_output += [lst_ary_OneHot]
    assert len(lst_train_EC_input) == len(lst_train_DC_output), "??"
    # ary_train_EC_input = np.concatenate(lst_train_EC_input,axis=0).reshape(-1,80,4096)
    # ary_train_DC_output = np.concatenate(lst_train_DC_output,axis=0).reshape(-1,lang.max_len_seq,lang.n_words)
    print ('here')
    ary_train_EC_input = np.asarray(lst_train_EC_input).reshape(-1,80,4096)
    print ('go go go')
    del lst_train_EC_input
    ary_train_DC_output = np.asarray(lst_train_DC_output).reshape(-1,lang.max_len_seq,lang.n_words)
    del lst_train_DC_output

    print (ary_train_EC_input.shape)
    print (ary_train_DC_output.shape)

    ### add "<BOS>" to ary_train_DC_input
    ary_temp = np.zeros((len(ary_train_EC_input),1,lang.n_words))
    ary_temp[:,0,0] = 1
    if teacherForce :
        ary_train_DC_input = np.concatenate([ary_temp,ary_train_DC_output[:,:-1]],axis=1)
    else :
        ary_train_DC_input = ary_temp


### for testing data
if test_data_loading :
    with open('{}testing_label.json'.format(path_data)) as f :
        lst_dict_label_test = json.load(f)
    ### pair caption and vedio feature
    ### output : ary_test_EC_input
    lst_id_test = [dict_label_test['id'] for dict_label_test in lst_dict_label_test]

    ### just check
    lst_id_test_2 = []
    with open('{}{}.txt'.format(path_data, 'testing_id')) as f :
        for line in f.readlines() :
            lst_id_test_2 += [line.rstrip('\n')]
    for i in range(len(lst_id_test)) :
        assert str(lst_id_test[i]) == str(lst_id_test_2[i]), 'error here'

    lst_test_EC_input = []
    for i, id in enumerate(lst_id_test) :
        npy = np.load('{}testing_data/feat/{}.npy'.format(path_data, id))
        lst_test_EC_input += [npy]
    ary_test_EC_input = np.concatenate(lst_test_EC_input,axis=0).reshape(-1,80,4096)

    ary_temp = np.zeros((len(ary_test_EC_input),1,lang.n_words))
    ary_temp[:,0,0] = 1
    ary_test_DC_input = ary_temp
    
### for peer review
if peer_review_data_loading :
    pass


# In[10]:


def model_pretrain(lang=lang) :

    EC_input = Input(shape=(80,4096))
    EC_output = GRU(32,return_state=False, return_sequences=True, activation='selu')(EC_input)
    EC_output, EC_output_state = GRU(32,return_state=True, activation='selu')(EC_output)
    DC_input = Input(shape=(None,lang.n_words))
    DC_input_M = Masking(mask_value=0.0)(DC_input)

    DC_gru1 = GRU(32, return_sequences=True, activation='selu')
    DC_time_dense = TimeDistributed(Dense(lang.n_words, activation='softmax'))

    DC_output = DC_gru1(DC_input_M, initial_state=EC_output_state)
    DC_output = DC_time_dense(DC_output)

    model = Model([EC_input,DC_input],DC_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print (model.summary())
    return model


# In[11]:


def model_DC_only1(lang=lang) :
    EC_input = Input(shape=(80,4096))
    EC_output = GRU(32,return_state=False, return_sequences=True, activation='selu')(EC_input)
    EC_output, EC_output_state = GRU(32,return_state=True, activation='selu')(EC_output)
    DC_input = Input(shape=(1,lang.n_words))
    DC_input_M = Masking(mask_value=0.0)(DC_input)

    DC_gru1 = GRU(32, return_sequences=True, return_state=True, activation='selu')
    DC_time_dense = TimeDistributed(Dense(lang.n_words, activation='softmax'))

    DC_output_state = EC_output_state
    DC_output = DC_input_M
    lst_DC_output = []
    for _ in range(lang.max_len_seq) :
        DC_output, DC_output_state = DC_gru1(DC_output, initial_state=DC_output_state)
        DC_output = DC_time_dense(DC_output)
        lst_DC_output += [DC_output]

    DC_output = Lambda(lambda x: K.concatenate(x, axis=1))(lst_DC_output)

    model = Model([EC_input,DC_input],DC_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print (model.summary())
    return model


# In[12]:


# def model_DC_only1(lang=lang) :
#     EC_input = Input(shape=(80,4096))
#     EC_output = GRU(64,return_state=False, return_sequences=True, activation='selu')(EC_input)
#     EC_output = BatchNormalization()(EC_output)
#     EC_output = GRU(64,return_state=False, return_sequences=True, activation='selu')(EC_output)
#     EC_output = BatchNormalization()(EC_output)
#     EC_output, EC_output_state = GRU(64,return_state=True, activation='selu')(EC_output)
#     EC_output_stage = BatchNormalization()(EC_output_state)
#     DC_input = Input(shape=(1,lang.n_words))
# #     DC_input_M = Masking(mask_value=0.0)(DC_input)

#     DC_gru1 = GRU(64, return_sequences=True, return_state=True, activation='selu')
#     DC_gru2 = GRU(64, return_sequences=True, return_state=True, activation='selu')
#     DC_gru3 = GRU(64, return_sequences=True, return_state=True, activation='selu')
#     DC_time_dense = TimeDistributed(Dense(lang.n_words, activation='softmax'))
#     Ba_output_1 = TimeDistributed(BatchNormalization())
#     Ba_output_2 = TimeDistributed(BatchNormalization())
#     Ba_output_3 = TimeDistributed(BatchNormalization())

#     DC_output_state1 = EC_output_state
#     DC_output_state2 = EC_output_state
#     DC_output_state3 = EC_output_state
#     DC_output = DC_input
#     lst_DC_output = []
#     for _ in range(lang.max_len_seq) :
#         DC_output, DC_output_state1 = DC_gru1(DC_output, initial_state=DC_output_state1)
#         DC_output = Ba_output_1(DC_output)
#         DC_output, DC_output_state2 = DC_gru2(DC_output, initial_state=DC_output_state2)
#         DC_output = Ba_output_2(DC_output)
#         DC_output, DC_output_state3 = DC_gru3(DC_output, initial_state=DC_output_state3)
#         DC_output = Ba_output_3(DC_output)
#         DC_output = DC_time_dense(DC_output)
#         lst_DC_output += [DC_output]

#     DC_output = Lambda(lambda x: K.concatenate(x, axis=1))(lst_DC_output)

#     model = Model([EC_input,DC_input],DC_output)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
#     print (model.summary())
#     return model


# In[13]:


def ary_pred_to_sentence(ary_pred) :
    ary_pred_argmax = np.argmax(ary_pred,axis=2)
    lst_ans_numbers = ary_pred_argmax.tolist()
    lst_ans_string = []
    for ans_numbers in lst_ans_numbers :
        lst_ans_string += [' '.join([lang.index2word[ans] for ans in ans_numbers])]
    return lst_ans_string


# In[17]:


### main
# MCP = keras.callbacks.ModelCheckpoint('./model/{}_{}.h5'.format(model_name,k), monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
# ES = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=0, mode='auto')

if teacherForce :
    model = model_pretrain(lang)
else :
    model = model_DC_only1(lang)
    if do_training :
        ary_temp = np.zeros((len(ary_train_EC_input),1,lang.n_words))
        ary_temp[:,0,0] = 1
        ary_train_DC_input = ary_temp
    
    
if loading_model :
    print ('loading and seting weight...')
    if special_task :
        with open('./model_weight/lst_layer_weights_special.pkl'.format(i), "rb") as f:
            lst_layer_weights = pickle.load(f)
    elif teacherForce :
        with open('./weights/lst_layer_weights.pkl'.format(i), "rb") as f:
            lst_layer_weights = pickle.load(f)
    else :
#         with open('./weights/lst_layer_weights_noTeacher_special.pkl'.format(i), "rb") as f:
        with open('./weights/lst_layer_weights.pkl'.format(i), "rb") as f:
            lst_layer_weights = pickle.load(f)
    for i, layer in enumerate(model.layers) :
        if i > 6 :
            break
        print (i)
        layer.set_weights(lst_layer_weights[i])

if do_training :
    log = model.fit([ary_train_EC_input, ary_train_DC_input],ary_train_DC_output, epochs=100, batch_size=128, validation_split=0., callbacks=[]) 
    df_log = log.history
    #fig = keras_log_plot(df_log)
    
if save_model :
    print ('saving model...')
    if not os.path.isdir('./model_weight') :
        os.mkdir('./model_weight')
    k = 0
    while 1 :
        if os.path.isfile('./model_weight/lst_layer_weights_{}.h5'.format(k)) :
            k += 1
        else :
            break
    lst_weights = []
    for i, layer in enumerate(model.layers) :
        lst_weights += [layer.get_weights()] # list of numpy arrays
        #np.save(ary_weights,'./weight/layer{}'.format(i))
    with open('./model_weight/lst_layer_weights_{}.h5'.format(k), "wb") as f:
        pickle.dump(lst_weights,f)



# In[18]:


### prediction

### train data prediction
if train_data_loading :
    if teacherForce :
        ary_pred_train = model.predict([ary_train_EC_input, ary_train_DC_input])
    else :
        ary_temp = np.zeros((len(ary_train_EC_input),1,lang.n_words))
        ary_temp[:,0,0] = 1
        ary_pred_train = model.predict([ary_train_EC_input, ary_temp])
    ary_pred_argmax_train = np.argmax(ary_pred_train,axis=2)
    print (ary_pred_train.shape)
    print (ary_pred_argmax_train.shape)
    print ('\n')

    lst_ans_string_train = ary_pred_to_sentence(ary_pred_train)
    print ('train : ')
    print (lst_ans_string_train[:5])

### testing data prediction
# model_test = model_DC_only1(lang)
# ary_pred_test = model_test.predict([ary_test_EC_input, ary_test_DC_input])
ary_pred_test = model.predict([ary_test_EC_input, ary_test_DC_input])

lst_ans_string_test = ary_pred_to_sentence(ary_pred_test)
print ('test : ')
print (lst_ans_string_test[:5])
print (lst_id_test[:5])





# In[19]:


# ary_sample_output_testset = np.loadtxt(open(path_data + "sample_output_testset.txt", "rb"), delimiter=",")
# for i, name in enumerate(ary_sample_output_testset[:][0]) :
#     print (name)

### for special task
if special_task :
    lst_id_special = ['klteYv1Uv9A_27_33.avi','5YJaS2Eswg0_22_26.avi','UbmZAe5u5FI_132_141.avi','JntMAcTlOF0_50_70.avi','tJHUH9tpqPg_113_118.avi']
    lst_ans_pair = []
    for i,id in enumerate(lst_id_test) :
        if id in lst_id_special :
            ans = lst_ans_string_test[i].split(' <EOS>')[0]
            lst_ans_pair += [[id,ans]]
    
    lst_ans_pair_sort = []

    for id in lst_id_special :
        for ans_pair in lst_ans_pair :
            if id == ans_pair[0] :
                lst_ans_pair_sort += [ans_pair]
    df_ans = pd.DataFrame(lst_ans_pair_sort)
    print (df_ans)
    df_ans.to_csv("./{}".format(output_name), index=False, header=False)
    

