
# coding: utf-8

# In[1]:


import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
import pickle

import skimage
import skimage.io
import skimage.transform
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image
import scipy
from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
def leaky_relu(features, alpha=0.2, name=None):
    with ops.name_scope(name, "LeakyRelu", [features, alpha]):
        features = ops.convert_to_tensor(features, name="features")
        alpha = ops.convert_to_tensor(alpha, name="alpha")
        return math_ops.maximum(alpha * features, features)


# ## argument setting

# In[2]:


random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

img_path = './data'
len_img_all = 33431
condition = True
load_data = True
tag_img_only = True
batch_size = 16
iteration = 100000
D_ITER = 1
G_ITER = 1
d_lda = 10

toy_test = False
if toy_test :
    len_img_all = 512
    load_data = False
if tag_img_only :
    len_img_all = 11568
output_str = 'gan'
load_model = False

pretrain_hair = True
pretrain_eyes = True


# ## preprocessing

# In[3]:


def img_prep(save=False, lst_null=None) :
    print ('img_prep is runing')
    if tag_img_only :
        lst_img = []
        for img_i in range(len_img_all) :
            if img_i in lst_null :
                continue
            img_temp = skimage.io.imread('{}/faces/{}.jpg'.format(img_path,img_i))
            img_temp = skimage.transform.resize(img_temp, (64,64))
            lst_img += [img_temp]
    else :
        ary_img = np.zeros((len_img_all,64,64,3))
        for img_i in range(len_img_all) :
            img_temp = skimage.io.imread('{}/faces/{}.jpg'.format(img_path,img_i))
            img_temp = skimage.transform.resize(img_temp, (64,64))
            ary_img[img_i] = img_temp
    if save :
        if tag_img_only :
            print ('./data_pp/ary_img_withtag is saving')
            ary_img = np.asarray(lst_img)
            np.save('./data_pp/ary_img_withtag', ary_img)
        else :
            np.save('./data_pp/ary_img', ary_img)
            
    print ('len_ary_img : {}'.format(len(ary_img)))
    return ary_img
    
def tag_prep(save=False) :
    # 1,6
    # 2,4,12
    # 3,5,9
    # 7,10
    lst_hair = ['null', 'orange hair', 'white hair', 'aqua hair', 'gray hair',
                'green hair', 'red hair', 'purple hair', 'pink hair',
                'blue hair', 'black hair', 'brown hair', 'blonde hair']
    lst_eyes = ['null', 'gray eyes', 'black eyes', 'orange eyes',
                'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
                'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
    
    h_n = 0
    e_n = 0  
    lst_null = []
    if tag_img_only :
        lst_tag_hair = []
        lst_tag_eyes = []
        with open('{}/tags_clean.csv'.format(img_path), 'r') as f :
            for line_i,line in enumerate(f.readlines()[:len_img_all]) :
        #         sys.stdout.write("\r{}\t".format(line_i))
        #         sys.stdout.flush()

                FLAG_h = False
                FLAG_e = False
                FLAG_double_tag = False
                for h_i,h in enumerate(lst_hair) :
                    if h in line :
                        if FLAG_h :
                            # double tag
                            lst_tag_hair = lst_tag_hair[:-1]
                            lst_null += [line_i]
                            FLAG_double_tag = True
                            break
                        FLAG_h = 1
                        lst_tag_hair += [h_i]
                        h_n += 1
                if not FLAG_h :
                    lst_null += [line_i]
                    continue
                if FLAG_double_tag :
                    continue
                for e_i,e in enumerate(lst_eyes) :
                    if e in line :
                        if FLAG_e :
                            # double tag
                            lst_tag_eyes = lst_tag_eyes[:-1]
                            lst_null += [line_i]
                            lst_tag_hair = lst_tag_hair[:-1]
                            break
                        FLAG_e = 1
                        lst_tag_eyes += [e_i]
                        e_n += 1
                if not FLAG_e :
                    lst_null += [line_i]
                    lst_tag_hair = lst_tag_hair[:-1]
            print (len(lst_tag_hair))
            print (len(lst_tag_eyes))
            print ('number of hair tag : {}'.format(h_n))
            print ('number of eyes tag : {}'.format(e_n))
            print ('number of total img : {}'.format(line_i+1))
            print ('number of total both tag : {}'.format(len(lst_tag_hair)))
            print ('number of drop tag : {}'.format(len(lst_null)))
            assert len(lst_tag_hair) == len(lst_tag_eyes), 'error'
            print ('tag preprocessing done')
            
    else :
        ary_tag_hair = np.zeros((len_img_all))
        ary_tag_eyes = np.zeros((len_img_all))
        with open('{}/tags_clean.csv'.format(img_path), 'r') as f :
            for line_i,line in enumerate(f.readlines()[:len_img_all]) :
        #         sys.stdout.write("\r{}\t".format(line_i))
        #         sys.stdout.flush()

                flag_h = 0
                flag_e = 0
                for h_i,h in enumerate(lst_hair) :
                    if h in line :
                        ary_tag_hair[line_i] = h_i
                        h_n += 1
                for e_i,e in enumerate(lst_eyes) :
                    if e in line :
                        ary_tag_eyes[line_i] = e_i
                        e_n += 1
            assert len(ary_tag_hair) == len(ary_tag_eyes), 'error'
            print ('number of hair tag : {}'.format(h_n))
            print ('number of eyes tag : {}'.format(e_n))
            print ('number of total img : {}'.format(line_i))
            print ('tag preprocessing done')
        
    if save :
        if not os.path.isdir('./data_pp') :
            os.makedirs('./data_pp')
        if tag_img_only :
            print ('./data_pp/ary_tag_hair_withtag is saving')
            ary_tag_hair = np.asarray(lst_tag_hair)
            ary_tag_eyes = np.asarray(lst_tag_eyes)
            np.save('./data_pp/ary_tag_hair_withtag', ary_tag_hair)
            np.save('./data_pp/ary_tag_eyes_withtag', ary_tag_eyes)
        else :
            print ('./data_pp/ary_tag_hair is saving')
            np.save('./data_pp/ary_tag_hair', ary_tag_hair)
            np.save('./data_pp/ary_tag_eyes', ary_tag_eyes)
    
    return ary_tag_hair, ary_tag_eyes, lst_null
    
def preprocess(show=True, save=False, load=False) :
    if load :
        if tag_img_only :
            ary_tag_hair = np.load('./data_pp/ary_tag_hair_withtag.npy')
            ary_tag_eyes = np.load('./data_pp/ary_tag_eyes_withtag.npy')
            ary_img = np.load('./data_pp/ary_img_withtag.npy')
        else :
            ary_tag_hair = np.load('./data_pp/ary_tag_hair.npy')
            ary_tag_eyes = np.load('./data_pp/ary_tag_eyes.npy')
            ary_img = np.load('./data_pp/ary_img.npy')
    else :
        ary_tag_hair, ary_tag_eyes, lst_null = tag_prep(save=save)
        ary_img = img_prep(save=save, lst_null=lst_null)

    if show :
        print ('ary_tag_hair shape : {}'.format(ary_tag_hair.shape))
        print ('ary_tag_eyes shape : {}'.format(ary_tag_eyes.shape))
        print ('ary_img shape : {}'.format(ary_img.shape))
#         imshow(ary_img[1])
    
    return ary_tag_hair, ary_tag_eyes, ary_img

ary_tag_hair, ary_tag_eyes, ary_img = preprocess(show=True,save=False,load=load_data)


# ## model

# In[ ]:


class GAN(object):
    def __init__(self) :
        self.lr = 0.0001
        self.lr_pre = 0.0001
        self.momentum = 0.5
        self.bs = batch_size # batch size is m of paper
        self.bs_pre = 128
        self.epoch = 10000
        self.hair_n = 13
        self.eyes_n = 12
        self.lda = d_lda
        self.epsilon = 0.5
        self.activation = leaky_relu
        self.initializer = tf.contrib.keras.initializers.he_normal()
            
    def build_G_net(self) :
        with tf.variable_scope('G') as g_scope:
            self.G_in_hair = tf.placeholder(tf.int32, shape=[None])
            self.G_in_eyes = tf.placeholder(tf.int32, shape=[None])
            self.G_in_noise = tf.placeholder(tf.float32, shape=[None,100])
            
            self.G_H_onehot = tf.one_hot(self.G_in_hair, self.hair_n)
            self.G_E_onehot = tf.one_hot(self.G_in_eyes, self.eyes_n)
            g = tf.concat([self.G_H_onehot, self.G_E_onehot, self.G_in_noise], axis=1)
            g = tf.layers.dense(g,4*4*1024,activation=None)
            g = tf.reshape(g,(-1,4,4,1024))
            
            g = tf.layers.conv2d_transpose(g, filters=512, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                           padding='same', activation=self.activation)
            g = tf.layers.batch_normalization(g)
            g = tf.layers.conv2d_transpose(g, filters=256, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                           padding='same', activation=self.activation)
            g = tf.layers.batch_normalization(g)
            g = tf.layers.conv2d_transpose(g, filters=128, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                           padding='same', activation=self.activation)
            g = tf.layers.batch_normalization(g)
            self.g = tf.layers.conv2d_transpose(g, filters=3, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                           padding='same', activation=tf.nn.tanh)
            self.img = (self.g + 1.0) / 2.0
        
    def build_D_net(self) :

        with tf.variable_scope('D') as d_scope:

            ### include right and fake1
            self.D_in_hair = tf.placeholder(tf.int32, shape=[None])
            self.D_in_eyes = tf.placeholder(tf.int32, shape=[None])
            self.D_in_img = tf.placeholder(tf.float32, shape=[None,64,64,3])
#             self.D_in_img = self.D_in_img*2 - 1
            
            D_H_onehot = tf.one_hot(self.D_in_hair, self.hair_n)
            D_E_onehot = tf.one_hot(self.D_in_eyes, self.eyes_n)
            
            self.D_in_hair_right = D_H_onehot[:self.bs]
            self.D_in_eyes_right = D_E_onehot[:self.bs]
            self.D_in_img_right = self.D_in_img[:self.bs]
            
            self.D_in_hair_fake = D_H_onehot[self.bs:]
            self.D_in_eyes_fake = D_E_onehot[self.bs:]
            self.D_in_img_fake = self.D_in_img[self.bs:]
            
            use_epsilon_uniform = 1
            if use_epsilon_uniform :
                self.epsilon = tf.placeholder(tf.float32, shape=[1])
                self.D_x_hat_hair = self.epsilon*self.D_in_hair_right + (1-self.epsilon)*self.D_in_hair_fake
                self.D_x_hat_eyes = self.epsilon*self.D_in_eyes_right + (1-self.epsilon)*self.D_in_eyes_fake
                self.D_x_hat_img = self.epsilon*self.D_in_img_right + (1-self.epsilon)*self.D_in_img_fake
            else :
                self.D_x_hat_hair = self.epsilon*self.D_in_hair_right + (1-self.epsilon)*self.D_in_hair_fake
                self.D_x_hat_eyes = self.epsilon*self.D_in_eyes_right + (1-self.epsilon)*self.D_in_eyes_fake
                self.D_x_hat_img = self.epsilon*self.D_in_img_right + (1-self.epsilon)*self.D_in_img_fake
                
            self.D_img = tf.concat([self.D_in_img_right,self.D_in_img_fake,self.D_x_hat_img], axis=0)
            self.D_hair = tf.concat([self.D_in_hair_right,self.D_in_hair_fake,self.D_x_hat_hair], axis=0)
            self.D_eyes = tf.concat([self.D_in_eyes_right,self.D_in_eyes_fake,self.D_x_hat_eyes], axis=0)
            
            d = tf.layers.conv2d(self.D_img, filters=128, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='d_c1', reuse=None)
            d = tf.layers.conv2d(d, filters=256, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='d_c2', reuse=None)
            d = tf.layers.conv2d(d, filters=512, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='d_c3', reuse=None)
            d = tf.layers.conv2d(d, filters=1024, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='d_c4', reuse=None)
            
            D_tag_in = tf.concat([self.D_hair, self.D_eyes], axis=1)
            D_tag_in = tf.layers.dense(D_tag_in,4*4*1024,activation=self.activation, name='d_d1', kernel_initializer=self.initializer, reuse=None)
            D_tag_in = tf.reshape(D_tag_in,(-1,4,4,1024))
            
            d = tf.concat([d, D_tag_in], axis=3)
            d = tf.layers.conv2d(d, filters=256, kernel_size=(1,1), strides=(1,1), 
                                 padding='same', activation=self.activation, name='d_c5', kernel_initializer=self.initializer, reuse=None)
            d = tf.reshape(d, [-1, 4*4*256]) 
            self.d_temp = tf.layers.dense(d, 1 ,activation=None, name='d_d2', kernel_initializer=self.initializer, reuse=None)
        
            self.d_right, self.d_fake, self.d_x_hat = tf.split(self.d_temp, num_or_size_splits=3, axis=0)
#             self.d_right, self.d_fake_temp = tf.split(self.d_temp, num_or_size_splits=2, axis=0)
#             self.d_fake, self.d_x_hat = tf.split(self.d_fake_temp, num_or_size_splits=2, axis=0)
            
            ### gradient penalty
#             gradient_temp = tf.gradients(self.d_x_hat,[self.D_x_hat_img,self.D_x_hat_hair,self.D_x_hat_eyes])
            gradient_img_temp = tf.gradients(self.d_x_hat,self.D_x_hat_img)
            gradient_hair_temp = tf.gradients(self.d_x_hat,self.D_x_hat_hair)
            gradient_eyes_temp = tf.gradients(self.d_x_hat,self.D_x_hat_eyes)
#             self.gradient_penalty = tf.maximum(tf.constant(0.0, shape=[self.bs,1]), tf.sqrt(tf.square(gradient_temp)) - tf.constant(1.0, shape=[self.bs,1]))
            self.gradient_penalty_img = tf.square(tf.sqrt(tf.reduce_sum(tf.square(gradient_img_temp), axis=1)) - 1)
#             self.gradient_penalty_img = self.lda * tf.reduce_mean(self.gradient_penalty_img)
            self.gradient_penalty_hair = tf.square(tf.sqrt(tf.reduce_sum(tf.square(gradient_hair_temp), axis=1)) - 1)
#             self.gradient_penalty_hair = self.lda * tf.reduce_mean(self.gradient_penalty_hair)
            self.gradient_penalty_eyes = tf.square(tf.sqrt(tf.reduce_sum(tf.square(gradient_eyes_temp), axis=1)) - 1)
#             self.gradient_penalty_eyes = self.lda * tf.reduce_mean(self.gradient_penalty_eyes)
            self.gradient_penalty = self.lda * (tf.reduce_mean(self.gradient_penalty_img)
                                                + tf.reduce_mean(self.gradient_penalty_hair)/10
                                                + tf.reduce_mean(self.gradient_penalty_eyes)/10)
            
            ### final loss
            self.d_loss = -(tf.reduce_mean(self.d_right) - tf.reduce_mean(self.d_fake) - self.gradient_penalty)
            self.d_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"D/")
            self.d_train = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=0.0,beta2=0.9).minimize(self.d_loss, var_list=self.d_train_vars)
            
            ### G_D loss
            self.G_D_in_hair = self.G_in_hair
            self.G_D_in_eyes = self.G_in_eyes
            self.G_D_in_img = self.g
            
            self.G_D_eyes = tf.one_hot(self.G_D_in_eyes, self.eyes_n)
            self.G_D_hair = tf.one_hot(self.G_D_in_hair, self.hair_n)
            
            g_d = tf.layers.conv2d(self.G_D_in_img, filters=128, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='d_c1', reuse=True)
            g_d = tf.layers.conv2d(g_d, filters=256, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='d_c2', reuse=True)
            g_d = tf.layers.conv2d(g_d, filters=512, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='d_c3', reuse=True)
            g_d = tf.layers.conv2d(g_d, filters=1024, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='d_c4', reuse=True)
            
            G_D_tag_in = tf.concat([self.G_D_hair, self.G_D_eyes], axis=1)
            G_D_tag_in = tf.layers.dense(G_D_tag_in,4*4*1024,activation=self.activation, name='d_d1', kernel_initializer=self.initializer, reuse=True)
            G_D_tag_in = tf.reshape(G_D_tag_in,(-1,4,4,1024))
            
            g_d = tf.concat([g_d, G_D_tag_in], axis=3)
            g_d = tf.layers.conv2d(g_d, filters=256, kernel_size=(1,1), strides=(1,1), 
                                 padding='same', activation=self.activation, name='d_c5', kernel_initializer=self.initializer, reuse=True)
            g_d = tf.reshape(g_d, [-1, 4*4*256]) 
            self.g_d_temp = tf.layers.dense(g_d, 1 ,activation=None, name='d_d2', kernel_initializer=self.initializer, reuse=True)
            
            self.gd_loss = - tf.reduce_mean(self.g_d_temp)
            
            #
            # HE classification
            #
            
            # hair classification pretrained
            self.hair_in_hair = tf.placeholder(tf.int32, shape=[None])
            self.hair_in_img = tf.placeholder(tf.float32, shape=[None,64,64,3]) 

            hair_H_onehot = tf.one_hot(self.hair_in_hair, self.hair_n)
            hair_temp = tf.layers.batch_normalization(self.hair_in_img, name='hair_b0', reuse=None)
            hair_temp = tf.layers.conv2d(hair_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='hair_c1', reuse=None)
            hair_temp = tf.layers.batch_normalization(hair_temp, name='hair_b1', reuse=None)
            hair_temp = tf.layers.conv2d(hair_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='hair_c2', reuse=None)
            hair_temp = tf.layers.batch_normalization(hair_temp, name='hair_b2', reuse=None)
            hair_temp = tf.layers.conv2d(hair_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='hair_c3', reuse=None)
            hair_temp = tf.layers.batch_normalization(hair_temp, name='hair_b3', reuse=None)
            hair_temp = tf.layers.conv2d(hair_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='hair_c4', reuse=None)
            hair_temp = tf.reshape(hair_temp, [-1, 4*4*32]) 
            self.hair = tf.layers.dense(hair_temp, self.hair_n ,activation=tf.nn.softmax, name='hair_s2', kernel_initializer=self.initializer, reuse=None)

            self.hair_loss = tf.reduce_mean(-tf.reduce_sum(hair_H_onehot * tf.log(self.hair), axis=1))
            self.hair_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"D/")
            self.hair_train = tf.train.AdamOptimizer(learning_rate=self.lr_pre).minimize(self.hair_loss, var_list=self.hair_train_vars)
            
            # eyes classification pretrained
            self.eyes_in_eyes = tf.placeholder(tf.int32, shape=[None])
            self.eyes_in_img = tf.placeholder(tf.float32, shape=[None,64,64,3]) 

            eyes_E_onehot = tf.one_hot(self.eyes_in_eyes, self.eyes_n)
            eyes_temp = tf.layers.batch_normalization(self.eyes_in_img, name='eyes_b0', reuse=None)
            eyes_temp = tf.layers.conv2d(eyes_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='eyes_c1', reuse=None)
            eyes_temp = tf.layers.batch_normalization(eyes_temp, name='eyes_b1', reuse=None)
            eyes_temp = tf.layers.conv2d(eyes_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='eyes_c2', reuse=None)
            eyes_temp = tf.layers.batch_normalization(eyes_temp, name='eyes_b2', reuse=None)
            eyes_temp = tf.layers.conv2d(eyes_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='eyes_c3', reuse=None)
            eyes_temp = tf.layers.batch_normalization(eyes_temp, name='eyes_b3', reuse=None)
            eyes_temp = tf.layers.conv2d(eyes_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='eyes_c4', reuse=None)
            eyes_temp = tf.reshape(eyes_temp, [-1, 4*4*32]) 
            self.eyes = tf.layers.dense(eyes_temp, self.eyes_n ,activation=tf.nn.softmax, name='eyes_s2', kernel_initializer=self.initializer, reuse=None)

            self.eyes_loss = tf.reduce_mean(-tf.reduce_sum(eyes_E_onehot * tf.log(self.eyes), axis=1))
            self.eyes_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"D/")
            self.eyes_train = tf.train.AdamOptimizer(learning_rate=self.lr_pre).minimize(self.eyes_loss, var_list=self.eyes_train_vars)
            
            ### hair classification
            gh_temp = tf.layers.batch_normalization(self.g, name='hair_b0', reuse=True)
            gh_temp = tf.layers.conv2d(gh_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='hair_c1', reuse=True)
            gh_temp = tf.layers.batch_normalization(gh_temp, name='hair_b1', reuse=True)
            gh_temp = tf.layers.conv2d(gh_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='hair_c2', reuse=True)
            gh_temp = tf.layers.batch_normalization(gh_temp, name='hair_b2', reuse=True)
            gh_temp = tf.layers.conv2d(gh_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='hair_c3', reuse=True)
            gh_temp = tf.layers.batch_normalization(gh_temp, name='hair_b3', reuse=True)
            gh_temp = tf.layers.conv2d(gh_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='hair_c4', reuse=True)
            gh_temp = tf.reshape(gh_temp, [-1, 4*4*32]) 
            self.gh = tf.layers.dense(gh_temp, self.hair_n ,activation=tf.nn.softmax, name='hair_s2', kernel_initializer=self.initializer, reuse=True)
            
            self.gh_loss = tf.reduce_mean(-tf.reduce_sum(self.G_H_onehot * tf.log(self.gh), axis=1))
            
            ### eyes classification
            ge_temp = tf.layers.batch_normalization(self.g, name='eyes_b0', reuse=True)
            ge_temp = tf.layers.conv2d(ge_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='eyes_c1', reuse=True)
            ge_temp = tf.layers.batch_normalization(ge_temp, name='eyes_b1', reuse=True)
            ge_temp = tf.layers.conv2d(ge_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='eyes_c2', reuse=True)
            ge_temp = tf.layers.batch_normalization(ge_temp, name='eyes_b2', reuse=True)
            ge_temp = tf.layers.conv2d(ge_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='eyes_c3', reuse=True)
            ge_temp = tf.layers.batch_normalization(ge_temp, name='eyes_b3', reuse=True)
            ge_temp = tf.layers.conv2d(ge_temp, filters=32, kernel_size=(5,5), strides=(2,2), kernel_initializer=self.initializer,
                                 padding='same', activation=self.activation, name='eyes_c4', reuse=True)
            ge_temp = tf.reshape(ge_temp, [-1, 4*4*32]) 
            self.ge = tf.layers.dense(ge_temp, self.eyes_n ,activation=tf.nn.softmax, name='eyes_s2', kernel_initializer=self.initializer, reuse=True)

            self.ge_loss = tf.reduce_mean(-tf.reduce_sum(self.G_E_onehot * tf.log(self.ge), axis=1))
            
            
            
            self.gd_loss_final = 1.*self.gd_loss + 1.*2.*self.gh_loss + 1.*5.*self.ge_loss
            
            self.gd_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"G/")
            self.gd_train = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=0.0,beta2=0.9).minimize(self.gd_loss_final, var_list=self.gd_train_vars)
            
    def build_net(self) :
#         self.build_hair_net()
#         self.build_eyes_net()
        
        self.build_G_net()
        self.build_D_net()
        
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.sum_img = tf.summary.image('image',self.img, max_outputs=5)
        self.sum_D_loss = tf.summary.scalar('D_loss',self.d_loss)
        self.sum_G_loss = tf.summary.scalar('G_loss',self.gd_loss_final)
        self.sum_merge = tf.summary.merge([self.sum_img, self.sum_D_loss, self.sum_G_loss]) 
        
        self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
#         self.writer = tf.summary.FileWriter("logs_temp/", tf.get_default_graph())
        self.writer.flush()


# ## training

# In[ ]:


def training() :
    tf.reset_default_graph()
    ary_tag_hair, ary_tag_eyes, ary_img = preprocess(save=False,load=load_data)
    if condition :
        gan = GAN()
        gan.build_net()
        if load_model :
            pass
        pretrain_hair = True
        if pretrain_hair :
            for i in range(14) :
                print (i)
                ary_tag_hair_sh, ary_tag_eyes_sh, ary_img_sh = shuffle(ary_tag_hair[:-1000], ary_tag_eyes[:-1000], ary_img[:-1000], random_state=i)
                ary_img_sh = ary_img_sh*2 -1
                b_i = 0
                while b_i+gan.bs_pre < len_img_all-1000 :
                    sys.stdout.write("\r{}\t".format(b_i))
                    sys.stdout.flush()
                    b_tag_hair = ary_tag_hair_sh[b_i:b_i+gan.bs_pre]
                    b_img = ary_img_sh[b_i:b_i+gan.bs_pre]
                    
                    _, loss = gan.sess.run([gan.hair_train,gan.hair_loss], feed_dict={gan.hair_in_hair:b_tag_hair,
                                                            gan.hair_in_img:b_img})
#                     print (loss)
                    b_i += gan.bs_pre
                ### validation    
                y_pred = np.argmax(gan.sess.run(gan.hair, feed_dict={gan.hair_in_img:ary_img[-500:]}), axis=1)
                y_true = ary_tag_hair[-500:]
#                 print ('\n')
#                 print (y_pred)
#                 print (y_true)
                print ('acc : {}'.format(accuracy_score(y_true,y_pred)))

        pretrain_eyes = True
        if pretrain_eyes :
            for i in range(14) :
                print (i)
                ary_tag_eyes_sh, ary_tag_eyes_sh, ary_img_sh = shuffle(ary_tag_eyes[:-1000], ary_tag_eyes[:-1000], ary_img[:-1000], random_state=i)
                ary_img_sh = ary_img_sh*2 -1
                b_i = 0
                while b_i+gan.bs_pre < len_img_all-1000 :
                    sys.stdout.write("\r{}\t".format(b_i))
                    sys.stdout.flush()
                    b_tag_eyes = ary_tag_eyes_sh[b_i:b_i+gan.bs_pre]
                    b_img = ary_img_sh[b_i:b_i+gan.bs_pre]
                    
                    _, loss = gan.sess.run([gan.eyes_train,gan.eyes_loss], feed_dict={gan.eyes_in_eyes:b_tag_eyes,
                                                            gan.eyes_in_img:b_img})
#                     print (loss)
                    b_i += gan.bs_pre
                ### validation    
                y_pred = np.argmax(gan.sess.run(gan.eyes, feed_dict={gan.eyes_in_img:ary_img[-500:]}), axis=1)
                y_true = ary_tag_eyes[-500:]
#                 print ('\n')
#                 print (y_pred)
#                 print (y_true)
                print ('acc : {}'.format(accuracy_score(y_true,y_pred)))
    else :
        gan = GAN_no_condition()
        gan.build_net()
    
    lst_loss_his_d = []
    lst_loss_his_g = []
    for i in range(iteration) :
        print(i)
        ary_tag_hair_sh, ary_tag_eyes_sh, ary_img_sh = shuffle(ary_tag_hair, ary_tag_eyes, ary_img, random_state=i)
        gen_tag_hair = np.random.randint(13, size=int(len_img_all/2+1))
        gen_tag_eyes = np.random.randint(12, size=int(len_img_all/2+1))
        ary_img_sh = ary_img_sh*2 -1
        
        b_i = 0
        while b_i+gan.bs <= len_img_all :
            b_tag_hair_right = ary_tag_hair_sh[b_i:b_i+gan.bs]
            b_tag_eyes_right = ary_tag_eyes_sh[b_i:b_i+gan.bs]
            b_img_right = ary_img_sh[b_i:b_i+gan.bs]
            
            # fake1 (right img wrong text)
            b_tag_hair_fake1 = np.random.randint(1,13, size=int(gan.bs/4))
            b_tag_eyes_fake1 = np.random.randint(1,12, size=int(gan.bs/4))
            b_img_fake1 = np.copy(b_img_right[:int(gan.bs/4)])
            for ii in range(int(gan.bs/4)) :
                while b_tag_hair_fake1[ii] == b_tag_hair_right[ii] :
                    b_tag_hair_fake1[ii] = random.randint(0,13)
                while b_tag_eyes_fake1[ii] == b_tag_eyes_right[ii] :
                    b_tag_eyes_fake1[ii] = random.randint(0,12)
                    
            # fake2 (wrong img right text)
            b_tag_hair_fake2 = np.copy(b_tag_hair_right[:int(gan.bs/4)])
            b_tag_eyes_fake2 = np.copy(b_tag_eyes_right[:int(gan.bs/4)])
            lst_random_num = random.sample(range(len_img_all),k=int(gan.bs/4))
            b_img_fake2 = np.copy(ary_img_sh[lst_random_num])
            
            # fake3 (generate img right text)
            b_tag_hair_fake3 = gen_tag_hair[int(b_i/2):int(b_i/2+gan.bs/2)]
            b_tag_eyes_fake3 = gen_tag_eyes[int(b_i/2):int(b_i/2+gan.bs/2)]
            ary_temp = np.random.normal(0,1,[b_tag_eyes_fake3.shape[0],100])

            b_img_fake3 = gan.sess.run(gan.g, feed_dict={gan.G_in_hair:b_tag_hair_fake3, 
                                                         gan.G_in_eyes:b_tag_eyes_fake3, 
                                                         gan.G_in_noise:ary_temp})
            
            
            # update D
            b_tag_hair = np.concatenate((b_tag_hair_right,b_tag_hair_fake1,b_tag_hair_fake2,b_tag_hair_fake3), axis=0)
            b_tag_eyes = np.concatenate((b_tag_eyes_right,b_tag_eyes_fake1,b_tag_eyes_fake2,b_tag_eyes_fake3), axis=0)
            b_img = np.concatenate((b_img_right,b_img_fake1,b_img_fake2,b_img_fake3), axis=0)
            b_epsilon = np.random.rand(1,)
            for i2 in range(D_ITER) :
                _, loss_D = gan.sess.run([gan.d_train,gan.d_loss], feed_dict={gan.D_in_hair:b_tag_hair, 
                                                          gan.D_in_eyes:b_tag_eyes, 
                                                          gan.D_in_img:b_img,
                                                          gan.epsilon:b_epsilon})
                tf.summary.scalar('loss_D', loss_D)
            
                print ('D loss : {}'.format(loss_D))
            
            # update G
            b_tag_hair_g = gen_tag_hair[int(b_i/2):int(b_i/2+gan.bs/2)]
            b_tag_eyes_g = gen_tag_eyes[int(b_i/2):int(b_i/2+gan.bs/2)]
            ary_temp_g = np.random.normal(0,1,[b_tag_hair_g.shape[0],100])
            for i3 in range(G_ITER) :
                _, loss_G = gan.sess.run([gan.gd_train,gan.gd_loss], feed_dict={gan.G_in_hair:b_tag_hair_g, 
                                                         gan.G_in_eyes:b_tag_eyes_g, 
                                                         gan.G_in_noise:ary_temp_g})
    
                print ('G loss : {}'.format(loss_G))
            
            
            if b_i % 6400 == 0 :
                lst_loss_his_d += [loss_D]
                lst_loss_his_g += [loss_G]
                print ('saving model...')
                if not os.path.isdir('./model_tf') :
                    os.mkdir('./model_tf')
                if not os.path.isdir('./record') :
                    os.mkdir('./record')
                if not os.path.isdir('./img') :
                    os.mkdir('./img')
                k = 0
                while 1 :
                    if os.path.isfile('./model_tf/model_{}_{}.ckpt.meta'.format(output_str, k)) :
                        k += 1
                    else :
                        break
                save_path = gan.saver.save(gan.sess, './model_tf/model_{}_{}.ckpt'.format(output_str, k))
                with open('./record/loss_g_{}_{}.pkl'.format(output_str, k), 'wb') as f:
                    pickle.dump(lst_loss_his_g, f)
                with open('./record/loss_d_{}_{}.pkl'.format(output_str, k), 'wb') as f:
                    pickle.dump(lst_loss_his_d, f)
                img_sample = gan.sess.run(gan.g, feed_dict={gan.G_in_hair:b_tag_hair_fake3, 
                                                         gan.G_in_eyes:b_tag_eyes_fake3, 
                                                         gan.G_in_noise:ary_temp})
                scipy.misc.imsave('img/img_sample_{}_{}.jpg'.format(output_str,k),img_sample[0])
                summary = gan.sess.run(gan.sum_merge, feed_dict={gan.G_in_hair:b_tag_hair_fake3, 
                                                         gan.G_in_eyes:b_tag_eyes_fake3, 
                                                         gan.G_in_noise:ary_temp,
                                                         gan.D_in_hair:b_tag_hair, 
                                                         gan.D_in_eyes:b_tag_eyes, 
                                                         gan.D_in_img:b_img,
                                                         gan.epsilon:b_epsilon})
                gan.writer.add_summary(summary)
                gan.writer.flush()
                print("Model saved in file: %s" % save_path)
    
            b_i += gan.bs
        
training()    

