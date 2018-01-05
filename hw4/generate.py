
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
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow

from PIL import Image
import scipy
# from sklearn.utils import shuffle

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
def leaky_relu(features, alpha=0.2, name=None):
    with ops.name_scope(name, "LeakyRelu", [features, alpha]):
        features = ops.convert_to_tensor(features, name="features")
        alpha = ops.convert_to_tensor(alpha, name="alpha")
        return math_ops.maximum(alpha * features, features)


# ## argument setting

# In[7]:


random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

img_n = 5
batch_size = 16
d_lda = 10

if len(sys.argv) > 1 :
    text_path = sys.argv[1]
else :
    text_path = './data/testing_text.txt'


# In[3]:


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
            
            
            
            self.gd_loss_final = 1*self.gd_loss + 1*self.gh_loss + 1*self.ge_loss
            
            self.gd_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"G/")
            self.gd_train = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=0.0,beta2=0.9).minimize(self.gd_loss_final, var_list=self.gd_train_vars)
            
    def build_net(self) :
        self.build_G_net()
        self.build_D_net()
        
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        tf.summary.FileWriter("logs/", self.sess.graph)


# In[4]:


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


# In[42]:


def generate_img(testing_text_id,img_n=40,hair_style='null',eyes_style='null', gan=None) :
    seed_i = 3
    random.seed(seed_i)
    np.random.seed(seed_i)
    tf.set_random_seed(seed_i)
    lst_hair = ['null', 'orange hair', 'white hair', 'aqua hair', 'gray hair',
                'green hair', 'red hair', 'purple hair', 'pink hair',
                'blue hair', 'black hair', 'brown hair', 'blonde hair']
    lst_eyes = ['null', 'gray eyes', 'black eyes', 'orange eyes',
                'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
                'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
    
    hair_i = 0 
    eyes_i = 0
    for i,h in enumerate(lst_hair) :
        if h == hair_style :
            hair_i = i
    for i,e in enumerate(lst_eyes) :
        if e == eyes_style :
            eyes_i = i
            
#     if hair_i == 6 :
#         seed_i = 
    
    
    if hair_i != 0 :
        gen_tag_hair = np.zeros((int(img_n),), dtype=np.int) + hair_i
    else :
        gen_tag_hair = np.random.randint(1,13,size=img_n)
    if eyes_i != 0 :
        gen_tag_eyes = np.zeros((int(img_n),), dtype=np.int) + eyes_i
    else :
        gen_tag_eyes = np.random.randint(1,12,size=img_n)
    
    ary_temp = np.random.normal(0,1,[img_n,100])
    b_img = gan.sess.run(gan.g, feed_dict={gan.G_in_hair:gen_tag_hair, 
                                           gan.G_in_eyes:gen_tag_eyes, 
                                           gan.G_in_noise:ary_temp})
    b_img = (b_img + 1.0) / 2.0
        
    if not os.path.isdir('./samples') :
        os.makedirs('./samples')
    
    if testing_text_id == '1' :
        lst_img = b_img[[4,8,10,15,21]]
        for i,img in enumerate(lst_img) :
            scipy.misc.imsave('./samples/sample_{}_{}.jpg'.format(int(testing_text_id),i+1),img)
    if testing_text_id == '2' :
        lst_img = b_img[[7,20,22,29,32]]
        for i,img in enumerate(lst_img) :
            scipy.misc.imsave('./samples/sample_{}_{}.jpg'.format(int(testing_text_id),i+1),img)

    if testing_text_id == '3' :
        lst_img = b_img[[1,4,16,18,25]]
        for i,img in enumerate(lst_img) :
            scipy.misc.imsave('./samples/sample_{}_{}.jpg'.format(int(testing_text_id),i+1),img)

#     for i,img in enumerate(b_img) :
#         scipy.misc.imsave('./samples/sample_{}_{}.jpg'.format(int(testing_text_id),i+1),img)


# In[43]:


print (text_path)
tf.reset_default_graph()
gan = GAN()
gan.build_net()
gan.saver.restore(gan.sess, './model_tf/model_gan9_bs16_HEclass_original.ckpt')
# gan.saver.restore(gan.sess, './model_tf/model_gan9_bs16_HEclass_more.ckpt')
# gan.saver.restore(gan.sess, './model_tf/model_gan.ckpt')
with open(text_path, 'r') as f :
#     lst_text = f.readlines()
    for text in f.readlines() :
        lst_temp = text.split(',')
        testing_text_id = lst_temp[0]
        testing_text = lst_temp[1]
        hair_style, eyes_style = text_pre(text)
        generate_img(testing_text_id=testing_text_id,hair_style=hair_style,eyes_style=eyes_style,gan=gan)

