
# coding: utf-8

# In[1]:


import os
import sys
import time
import random
import numpy as np
# import pandas as pd
# import keras
import tensorflow as tf
import pickle

import skimage
import skimage.io
import skimage.transform
# from skimage.viewer import ImageViewer
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image
import scipy
from sklearn.utils import shuffle


# ## argument setting

# In[2]:


random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# def init():
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = tf.Session(config=config)
#     keras.backend.tensorflow_backend.set_session(session)
# init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

img_path = './data'
len_img_all = 33431
toy_test = True
load_data = True
if toy_test :
    len_img_all = 256
    load_data = False
tag_img_only = False
batch_size = 64
iteration = 1000


# ## preprocessing

# In[ ]:


def img_prep(save=False) :
    ary_img = np.zeros((len_img_all,64,64,3))
    for img_i in range(len_img_all) :
        img_temp = skimage.io.imread('{}/faces/{}.jpg'.format(img_path,img_i))
        img_temp = skimage.transform.resize(img_temp, (64,64))
        ary_img[img_i] = img_temp
    if save :
        np.save('./data_pp/ary_img', ary_img)
    return ary_img
    
def tag_prep(save=False) :
    lst_hair = ['null', 'orange hair', 'white hair', 'aqua hair', 'gray hair',
                'green hair', 'red hair', 'purple hair', 'pink hair',
                'blue hair', 'black hair', 'brown hair', 'blonde hair']
    lst_eyes = ['null', 'gray eyes', 'black eyes', 'orange eyes',
                'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
                'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
    
    ary_tag_hair = np.zeros((len_img_all))
    ary_tag_eyes = np.zeros((len_img_all))
    h_n = 0
    e_n = 0
    with open('{}/tags_clean.csv'.format(img_path), 'r') as f :
        for line_i,line in enumerate(f.readlines()[:len_img_all]) :
    #         sys.stdout.write("\r{}\t".format(line_i))
    #         sys.stdout.flush()

            for h_i,h in enumerate(lst_hair) :
                if h in line :
                    ary_tag_hair[line_i] = h_i
                    h_n += 1
            for e_i,e in enumerate(lst_eyes) :
                if e in line :
                    ary_tag_eyes[line_i] = e_i
                    e_n += 1
        print (h_n)
        print (e_n)
                
    if tag_img_only :
        tag_img_n = 0
        for i in range(len_img_all) :
            if ary_tag_hair[i] != 0 and ary_tag_eyes[i] != 0 :
                tag_img_n += 1
        print (tag_img_n)
        
    if save :
        if not os.path.isdir('./data_pp') :
            os.makedirs('./data_pp')
        np.save('./data_pp/ary_tag_hair', ary_tag_hair)
        np.save('./data_pp/ary_tag_eyes', ary_tag_eyes)
    return ary_tag_hair, ary_tag_eyes
    
def preprocess(show=True, save=False, load=False) :
    if load :
        ary_tag_hair = np.load('./data_pp/ary_tag_hair.npy')
        ary_tag_eyes = np.load('./data_pp/ary_tag_eyes.npy')
        ary_img = np.load('./data_pp/ary_img.npy')
    else :
        ary_tag_hair, ary_tag_eyes = tag_prep(save=save)
        ary_img = img_prep(save=save)

    if show :
        print ('ary_tag_hair shape : {}'.format(ary_tag_hair.shape))
        print ('ary_tag_eyes shape : {}'.format(ary_tag_eyes.shape))
        print ('ary_img shape : {}'.format(ary_img.shape))
        imshow(ary_img[1])
    
    return ary_tag_hair, ary_tag_eyes, ary_img
        # img = Image.open(img_path,'r')
# print (imshow(np.asarray(img)))
# img = image2tensor(img)
# print (imshow(np.asarray(img)))

# img = skimage.io.imread(img_path)
# img_resize = skimage.transform.resize(img, (64,64))
# print (img.shape)
# print (img_resize.shape)
# print (type(img))
# print (imshow(np.asarray(img_resize)))

# viewer = ImageViewer(img_path)
# viewer.show()
# image2tensor(img=img)
preprocess(save=False,load=load_data)


# ## model

# In[65]:


class GAN(object):
    def __init__(self) :
        self.lr = 0.0001
        self.momentum = 0.5
        self.bs = 32
        self.epoch = 300
        self.hair_n = 13
        self.eyes_n = 12
        self.lda = 10
        
    def build_G_net(self) :
        with tf.variable_scope('G') as s:
            self.G_in_hair = tf.placeholder(tf.int32, shape=[None])
            self.G_in_eyes = tf.placeholder(tf.int32, shape=[None])
            self.G_in_noise = tf.placeholder(tf.float32, shape=[None,100])
#             G_in_noise = tf.distributions.Normal(loc=0., scale=1.).sample([self.bs])
            
            G_H_onehot = tf.one_hot(self.G_in_hair, self.hair_n)
            G_E_onehot = tf.one_hot(self.G_in_eyes, self.eyes_n)
            g = tf.concat([G_H_onehot, G_E_onehot, self.G_in_noise], axis=1)
            
#             g = keras.layers.Dense(4*4*1024, activation='linear')(g)
            g = tf.layers.dense(g,4*4*1024,activation=None)
            g = tf.reshape(g,(-1,4,4,1024))
#             g = tf.nn.relu(g)
            
            g = tf.layers.conv2d_transpose(g, filters=512, kernel_size=(5,5), strides=(2,2), 
                                           padding='same', activation=tf.nn.relu)
            g = tf.layers.conv2d_transpose(g, filters=256, kernel_size=(5,5), strides=(2,2), 
                                           padding='same', activation=tf.nn.relu)
            g = tf.layers.conv2d_transpose(g, filters=128, kernel_size=(5,5), strides=(2,2), 
                                           padding='same', activation=tf.nn.relu)
            self.g = tf.layers.conv2d_transpose(g, filters=3, kernel_size=(5,5), strides=(2,2), 
                                           padding='same', activation=tf.nn.relu)
            
#         with tf.variable_scope('G_D') as s :
#             self.y_G_D = tf.placeholder(tf.float32, shape=[None,1])
            
#             d = tf.layers.conv2d(self.g, filters=128, kernel_size=(5,5), strides=(2,2), 
#                                  padding='same', activation=tf.nn.relu)
#             d = tf.layers.conv2d(d, filters=256, kernel_size=(5,5), strides=(2,2), 
#                                  padding='same', activation=tf.nn.relu)
#             d = tf.layers.conv2d(d, filters=512, kernel_size=(5,5), strides=(2,2), 
#                                  padding='same', activation=tf.nn.relu)
#             d = tf.layers.conv2d(d, filters=1024, kernel_size=(5,5), strides=(2,2), 
#                                  padding='same', activation=tf.nn.relu)
            
#             D_H_onehot = tf.one_hot(self.G_in_hair, self.hair_n)
#             D_E_onehot = tf.one_hot(self.G_in_eyes, self.eyes_n)
#             D_tag_in = tf.concat([D_H_onehot, D_E_onehot], axis=1)
#             D_tag_in = tf.layers.dense(D_tag_in,4*4*1024,activation=tf.nn.relu)
#             D_tag_in = tf.reshape(D_tag_in,(-1,4,4,1024))
            
#             d = tf.concat([d, D_tag_in], axis=3)
#             d = tf.layers.conv2d(d, filters=256, kernel_size=(1,1), strides=(1,1), 
#                                  padding='same', activation=tf.nn.relu)
# #             d = tf.layers.flatten(d)
#             d = tf.reshape(d, [-1, 4*4*256]) 
#             self.g_d = tf.layers.dense(d, 1 ,activation=None)
        
    def build_D_net(self) :
        with tf.variable_scope('D') as s:
            self.y_D = tf.placeholder(tf.float32, shape=[None,1])
            
            self.D_in_hair_right = tf.placeholder(tf.int32, shape=[None])
            self.D_in_eyes_right = tf.placeholder(tf.int32, shape=[None])
            self.D_in_img_right = tf.placeholder(tf.float32, shape=[None,64,64,3])
            
            
            
            
            self.D_in_hair = tf.concat(self.D_in_hair_right, sel)
            self.D_in_eyes_wrong = tf.placeholder(tf.int32, shape=[None])
            self.D_in_img_wrong = tf.placeholder(tf.float32, shape=[None,64,64,3])
            
            d = tf.layers.conv2d(self.D_in_img, filters=128, kernel_size=(5,5), strides=(2,2), 
                                 padding='same', activation=tf.nn.elu)
            d = tf.layers.conv2d(d, filters=256, kernel_size=(5,5), strides=(2,2), 
                                 padding='same', activation=tf.nn.elu)
            d = tf.layers.conv2d(d, filters=512, kernel_size=(5,5), strides=(2,2), 
                                 padding='same', activation=tf.nn.elu)
            d = tf.layers.conv2d(d, filters=1024, kernel_size=(5,5), strides=(2,2), 
                                 padding='same', activation=tf.nn.elu)
            
            D_H_onehot = tf.one_hot(self.D_in_hair, self.hair_n)
            D_E_onehot = tf.one_hot(self.D_in_eyes, self.eyes_n)
            D_tag_in = tf.concat([D_H_onehot, D_E_onehot], axis=1)
            D_tag_in = tf.layers.dense(D_tag_in,4*4*1024,activation=tf.nn.elu)
            D_tag_in = tf.reshape(D_tag_in,(-1,4,4,1024))
            
            d = tf.concat([d, D_tag_in], axis=3)
            d = tf.layers.conv2d(d, filters=256, kernel_size=(1,1), strides=(1,1), 
                                 padding='same', activation=tf.nn.elu)
#             d = tf.layers.flatten(d)
            d = tf.reshape(d, [-1, 4*4*256]) 
            self.d = tf.layers.dense(d, 1 ,activation=None)
        
            # loss
            ################################################################
            ############################# fix ##############################
            ################################################################
            tf.gradients(self.d, self.D_in_img)
            self.img_mean_gra = tf.placeholder(tf.float32, shape=[None,64,64,3]) # ?????? need to be fix
            print ('oo')
#             print (self.img_mean_gra)
#             print (self.img_mean_gra[0])
            self.img_mean_gra_resh = tf.reshape(self.img_mean_gra, (-1,64*64*3))
            gradient_penalty = self.lda * tf.square(tf.norm(self.img_mean_gra_resh, ord=2, axis=1) - 1.0)
#             print (tf.norm(self.img_mean_gra[0], ord=2).shape)
            print (gradient_penalty)
            
            self.d_right, self.d_wrong = tf.split(self.d, [self.bs,self.bs])
            # earth mover in 1d and gradient penalty
            self.d_loss = -(tf.reduce_mean(self.d_right) - tf.reduce_mean(self.d_wrong) ) + tf.reduce_mean(gradient_penalty)
#             self.d_loss = -self.d_loss
            self.d_train = tf.train.RMSPropOptimizer(self.lr).minimize(self.d_loss)
#             self.d_train = tf.train.AdamOptimizer(learning_rate=0.00001,beta1=0.0,beta2=0.9).minimize(self.d_loss)
            
    def build_net(self) :
        self.build_G_net()
        self.build_D_net()
        lst_G_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G/')
#         print (lst)
        lst = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_D/')
#         print (lst)
        lst_D_trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D/')
#         print (lst)
        
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        tf.summary.FileWriter("logs/", self.sess.graph)


# In[66]:


a = GAN()
a.build_net()


# ## training

# In[67]:


def training() :
    tf.reset_default_graph()
    ary_tag_hair, ary_tag_eyes, ary_img = preprocess(save=False,load=load_data)
    gan = GAN()
    gan.build_net()
    
    for i in range(iteration) :
        print(i)
        ary_tag_hair_sh, ary_tag_eyes_sh, ary_img_sh = shuffle(ary_tag_hair, ary_tag_eyes, ary_img, random_state=i)
#         lst_random_num = random.sample(range(len_img_all),k=gan.bs/2)
        gen_tag_hair = np.random.randint(13, size=int(len_img_all/2+1))
        gen_tag_eyes = np.random.randint(12, size=int(len_img_all/2+1))
        
        b_i = 0
        while b_i+gan.bs <= len_img_all :
            b_tag_hair_right = ary_tag_hair[b_i:b_i+gan.bs]
            b_tag_eyes_right = ary_tag_eyes[b_i:b_i+gan.bs]
            b_img_right = ary_img[b_i:b_i+gan.bs]
            
            
            # fake1 (right img wrong text)
            b_tag_hair_fake1 = np.random.randint(13, size=int(gan.bs/4))
            b_tag_eyes_fake1 = np.random.randint(12, size=int(gan.bs/4))
            b_img_fake1 = np.copy(b_img_right[:int(gan.bs/4)])
            for ii in range(int(gan.bs/4)) :
                if b_tag_hair_fake1[ii] == b_tag_hair_right[ii] :
                    b_tag_hair_fake1[ii] = random.randint(0,13)
                if b_tag_eyes_fake1[ii] == b_tag_eyes_right[ii] :
                    b_tag_eyes_fake1[ii] = random.randint(0,12)
                    
            # fake2 (wrong img right text)
            b_tag_hair_fake2 = np.copy(b_tag_hair_right[:int(gan.bs/4)])
            b_tag_eyes_fake2 = np.copy(b_tag_eyes_right[:int(gan.bs/4)])
            lst_random_num = random.sample(range(len_img_all),k=int(gan.bs/4))
            b_img_fake2 = np.copy(ary_img_sh[lst_random_num])
            
            # fake3 (generate img right text)
            dist = tf.distributions.Normal(loc=np.zeros(100).tolist(), scale=np.ones(100).tolist())
            temp = dist.sample([int(gan.bs/2)])
            ary_temp = gan.sess.run(temp)
            b_tag_hair_fake3 = gen_tag_hair[int(b_i/2):int(b_i/2+gan.bs/2)]
            b_tag_eyes_fake3 = gen_tag_eyes[int(b_i/2):int(b_i/2+gan.bs/2)]
#             print ('gg')
            b_img_fake3 = gan.sess.run(gan.g, feed_dict={gan.G_in_hair:b_tag_hair_fake3, 
                                                         gan.G_in_eyes:b_tag_eyes_fake3, 
                                                         gan.G_in_noise:ary_temp})
            
            
            # run D
            b_tag_hair = np.concatenate((b_tag_hair_right,b_tag_hair_fake1,b_tag_hair_fake2,b_tag_hair_fake3), axis=0)
            b_tag_eyes = np.concatenate((b_tag_eyes_right,b_tag_eyes_fake1,b_tag_eyes_fake2,b_tag_eyes_fake3), axis=0)
            b_img = np.concatenate((b_img_right,b_img_fake1,b_img_fake2,b_img_fake3), axis=0)
            
            img_right, img_worng = b_img[:gan.bs], b_img[gan.bs:]
            img_mean = (img_right + img_worng) / 2
            img_mean = np.concatenate([img_mean,img_mean], axis=0)
            
#             epsilon = tf.placeholder(tf.float32, shape=(batch_size, 1, 1, 1))
#             x_hat = epsilon * _x + (1.0 - epsilon) * _g_z
        
            # 1,64,64,64,3
            ################################################################
            ############################# fix ##############################
            ################################################################
#             epsilon = tf.placeholder(tf.float32, shape=(batch_size, 1, 1, 1))
#             x_hat = epsilon * _x + (1.0 - epsilon) * _g_z
            這個梯度根本沒有影響loss????
            img_mean_gra = gan.sess.run(tf.gradients(gan.d, gan.D_in_img), feed_dict={gan.D_in_hair:np.ones((64,)), 
                                                                                        gan.D_in_eyes:np.ones((64,)), 
                                                                                        gan.D_in_img:img_mean})

            img_mean_gra = img_mean_gra[0]#.reshape(64,64,64,3)
#             print (img_mean_gra[0])
#             print ('##############')
#             print (img_mean_gra[1])
            
            _, loss = gan.sess.run([gan.d_train,gan.d_loss], feed_dict={gan.D_in_hair:b_tag_hair, 
                                                          gan.D_in_eyes:b_tag_eyes, 
                                                          gan.D_in_img:b_img,
                                                          gan.img_mean_gra:img_mean_gra})
            
            print ('loss : {}'.format(loss))
        
        
            b_i += gan.bs
#             for ii in range(gan.bs/3) :
#                 if b_tag_hair_fake1[ii] == b_tag_hair_right[ii] :
#                     b_tag_hair_fake1[ii] = random.randint(0,13)
#                 if b_tag_eyes_fake1[ii] == b_tag_eyes_right[ii] :
#                     b_tag_eyes_fake1[ii] = random.randint(0,12)
                
#         b_tag_hair = ary_tag_hair[lst_random_num]
#         b_tag_eyes = ary_tag_eyes[lst_random_num]
#         b_img = ary_img[lst_random_num]
        
#         print (b_tag_hair.shape)
#         print (b_img.shape)
        
    
training()    


# In[13]:


def text_pre(text) :
    text = text.replace(',',' ')
    lst_text = text.split(' ')
    hair_i = 0
    eyes_i = 0
    for i,s in enumerate(lst_text) :
        if s == 'hair' :
            hair_i = i
        elif s == 'eyes' :
            eyes_i = i
    if hair_i :
        hair_style = lst_text[hair_i-1] + ' ' + lst_text[hair_i]
    else :
        hair_style = 'null'
    if eyes_i :
        eyes_style = lst_text[eyes_i-1] + ' ' + lst_text[eyes_i]
    else :
        eyes_style = 'null'
    
    return hair_style, eyes_style


# ## testing

# ## special task

# In[25]:


def generate_img() :
    for i in range(5) :
        ary_img = np.random.randint(255,size=(64,64,3))
        scipy.misc.imsave('sample_1_{}.jpg'.format(i+1),ary_img)


# In[26]:


text = 'red hair, green eyes'
# text = 'red eyes'
hair_style, eyes_style = text_pre(text)
print (hair_style)
print (eyes_style)

generate_img()


# In[7]:


start_t = time.time()
args = get_args()
model_type = 'CNN'
path_model = get_path_new_model(model_type)
#path_model = './model/CNN_1.h5'


print ('all process cost {} seconds'.format(time.time() - start_t))
print ('all done')


# In[ ]:


a = np.arange(15).reshape((3,5))
print (a[[0,2]])


# In[ ]:


# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
v = tf.constant(1, shape=[5,30])
v2 = tf.constant(2, shape=[5,30])
vv = tf.add(v,v2)
sess = tf.Session(config=config)
vv = sess.run(vv, feed_dict={})
print(vv)

print (v)
s0, s1, s2 = tf.split(v, [1, 1, 3], 0)
print (s0)
print (s2)


# In[19]:


a = np.arange(15).reshape((3,5))
b = np.ones((3,5))

print ((a+b)/2)

c = tf.placeholder(tf.float32, shape=(None,3,5))
c = tf.constant(1.0, shape=[3,5])
d = tf.norm(c, ord=2)
print (d)
sess = tf.Session(config=config)
vv = sess.run(d, feed_dict={})
print (vv)

