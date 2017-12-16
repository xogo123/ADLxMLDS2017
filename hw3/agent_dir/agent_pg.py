from agent_dir.agent import Agent
import scipy
import numpy as np


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        
        import os
        import sys
        import pickle
        import random
        import time
        import numpy as np
        import tensorflow as tf
        import keras
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)
        
        ### parameter ###
        best_model_name = ''
        
        self.args = args
        self.lr = self.args.learning_rate
        self.env = env
        self.output_str = 'pg_2a_elu_no_meanstd_a2c_sep_final'
        self.baseline = 0
#         self.gamma_pos = 1
#         self.gamma_neg = 0.9
        self.gamma = 0.99
        self.max_iter = 10000
        self.trajectory_n = int(1e7)
        self.action_n = 6
        self.ER = 0.05
        self.action2 = False
        if self.action2 :
            self.action_n = 2
        self.a2c_sep = True
        
#         def build_model():
#             self.x=tf.placeholder(tf.float32,shape=[None,6400])
#             self.v=tf.placeholder(tf.float32,shape=[None])
#             self.a=tf.placeholder(tf.int32,shape=[None])
#             temp = keras.layers.Dense(2048,activation='relu')(self.x)
#             self.a_prob = keras.layers.Dense(self.action_n,activation='softmax')(temp)
#             temp2 = keras.layers.Dense(2048,activation='relu')(self.x)
#             self.v_output = keras.layers.Dense(1,activation='linear')(temp2)
#             self.v_output = tf.reshape(self.v_output,[-1])
#             vloss=tf.reduce_sum(tf.square(self.v-self.v_output))
            
#             CE = tf.reduce_sum(-tf.log(self.a_prob)*tf.one_hot(self.a, self.action_n), axis=1)
#             self.obj = tf.reduce_sum(CE * self.v) 

#             self.v_train_op = tf.train.AdamOptimizer(self.lr).minimize(vloss)
#             self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.obj)
        def build_model() :
            self.input_s = tf.placeholder(tf.float32, shape=(None, 80, 80, 1))
            self.input_a = tf.placeholder(tf.int32, shape=(None,))
            self.input_R = tf.placeholder(tf.float32, shape=(None,1))
            
            with tf.variable_scope('pg') :
#                 c = keras.layers.Conv2D(16, (8,8), strides=(4,4), activation='relu')(self.input_s)
#                 c = keras.layers.Conv2D(32, (4,4), strides=(2,2), activation='relu')(c)
#                 c = keras.layers.Flatten()(c)
#                 c = keras.layers.Dense(128, activation='elu')(c)
                c = keras.layers.Flatten()(self.input_s)
                c = keras.layers.Dense(2048, activation='relu')(c)
                self.a_prob = keras.layers.Dense(self.action_n, activation='softmax')(c)
    
                # a2c
                c2 = c
                if self.a2c_sep :
#                     c2 = keras.layers.Conv2D(16, (8,8), strides=(4,4), activation='relu')(self.input_s)
#                     c2 = keras.layers.Conv2D(32, (4,4), strides=(2,2), activation='relu')(c2)
#                     c2 = keras.layers.Flatten()(c2)
#                     c2 = keras.layers.Dense(128, activation='elu')(c2)
                    c2 = keras.layers.Flatten()(self.input_s)
                    c2 = keras.layers.Dense(2048, activation='relu')(c2)
                self.v = keras.layers.Dense(1, activation='linear')(c2)
                self.v_loss = tf.reduce_sum(tf.square(self.input_R-self.v))
                self.v_train_op = tf.train.RMSPropOptimizer(self.lr, decay=0.9).minimize(self.v_loss)
                
#                 tf_mean, tf_variance= tf.nn.moments(self.input_R, [0], shift=None, name="reward_moments")
#                 tf_temp = (self.input_R - tf_mean) / tf.sqrt(tf_variance + 1e-6)
#                 tf_temp = (self.input_R - tf_mean - 0.2)
                
#                 a = tf.one_hot(self.input_a, self.action_n)
                C_E = -tf.reduce_sum(tf.log(self.a_prob)*tf.one_hot(self.input_a, self.action_n), axis=1, keep_dims=True)

#                 C_E = tf.multiply(C_E,tf_temp)
                C_E = tf.multiply(C_E,(self.input_R - self.v))
                self.obj = tf.reduce_sum(C_E)
#                 self.obj = tf.multiply(s,self.input_R)
                self.train_step = tf.train.RMSPropOptimizer(self.lr, decay=0.9).minimize(self.obj)
                
            
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            tf.summary.FileWriter("logs_pg/", self.sess.graph)
            
        build_model()

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
#             self.saver.restore(self.sess, './model_tf/model_pg_elu.ckpt')
            self.saver.restore(self.sess, './model_pg_elu_lr00025_2a_tanh_0.ckpt')
            print ('loading model finished...')

        ##################
        # YOUR CODE HERE #
        ##################
        pass
    

    
    def prepro(self,I):
        import numpy as np
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        I = I[::2,::2,0] # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1    # everything else (paddles, ball) just set to 1
#         return I.astype(np.float).ravel()
        return I.astype(np.float).reshape((80,80,1))
    '''
    def prepro(self,o,image_size=[80,80]):
        """
        Call this function to preprocess RGB image to grayscale image if necessary
        This preprocessing code is from
            https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py

        Input: 
        RGB image: np.array
            RGB screen of game, shape: (210, 160, 3)
        Default return: np.array 
            Grayscale image, shape: (80, 80, 1)

        """
        y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
        y = y.astype(np.uint8)
        resized = scipy.misc.imresize(y, image_size)
        return np.expand_dims(resized.astype(np.float32),axis=2)
    '''    
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.state_prev = np.zeros((80,80,1))
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        import os
        import sys
        import pickle
        import random
        import math
        import numpy as np
        import keras
        import tensorflow as tf
#         from collections import deque
        
        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        def r_decay_sum(lst_r) :
            r_sum = 0.
            gamma = self.gamma
            for r in lst_r:
                if r == 0 :
                    gamma *= self.gamma
                else :
                    gamma /= self.gamma
                    r_sum += gamma * r
                    gamma = self.gamma
            return r_sum - self.baseline
        
        def compute_R_decay(lst_r) :
            lst_r_decay = []
            gamma = 1
            r_temp = 0
            for r in reversed(lst_r):
                if r == 0 :
                    r_temp = r_temp * gamma
                    lst_r_decay += [r_temp]
                    gamma *= self.gamma
                else :
                    r_temp = r
                    lst_r_decay += [r]
                    gamma = self.gamma
            ary_r_decay = np.asarray(lst_r_decay[::-1]) # reverse
            return ary_r_decay

#             mean = np.mean(ary_r_decay)
#             stderr = np.std(ary_r_decay)
#             return (ary_r_decay - mean) / stderr
        
        lst_reward_episode = []
        lst_reward_his = []
        lst_loss_his = []
        for i0 in range(self.trajectory_n) :
            print ('trajectory : {}'.format(i0))
            lst_state = []
            lst_action = []
            lst_reward = []
            done = 0

            state_now = self.env.reset()
            state_now = self.prepro(state_now)
                
            reward = 100 # init not zero
            while not done :
                if reward :
                    state = state_now
                else :
                    state = state_now - state_prev

#                 action_p = self.sess.run(self.a_prob, feed_dict={self.input_s: state.reshape(1,80,80,1)}).ravel()
                action_p = self.sess.run(self.a_prob, feed_dict={self.input_s: state.reshape(1,6400)}).ravel()
                action = int(np.random.choice(self.action_n,p=action_p))
                lst_state += [state.tolist()]
                lst_action += [action]
                state_prev = state_now
                if self.action2 :
                    if action == 0 :
                        action = 2
                    elif action == 1 :
                        action = 5
                    else :
                        print ('error')
                        exit()
                state_now, reward, done, info = self.env.step(action)
                state_now = self.prepro(state_now)
                lst_reward += [reward]
            ary_s, ary_a = np.asarray(lst_state), np.asarray(lst_action)
            ary_R = compute_R_decay(lst_reward).reshape((-1,1))
            
            self.sess.run(self.v_train_op, feed_dict={self.input_s: ary_s,
                                                      self.input_a: ary_a,
                                                      self.input_R: ary_R})
            _, loss_temp = self.sess.run([self.train_step, self.obj], feed_dict={self.input_s: ary_s,
                                                                                 self.input_a: ary_a,
                                                                                 self.input_R: ary_R})
            print (loss_temp)
            sum_lst_reward = sum(lst_reward)
            if sum_lst_reward > -21 :
                print ('reward : {}'.format(sum_lst_reward))
            lst_reward_episode += [sum_lst_reward]
            ave_30_reward = sum(lst_reward_episode[-30:]) / float(len(lst_reward_episode[-30:]))
            print ('ave_30_reward : {}'.format(ave_30_reward))
            lst_reward_his += [ave_30_reward]
            lst_loss_his += [loss_temp]
                
            ## saving model ###
            if i0 % 30 == 29 :
                print ('saving model...')
                if not os.path.isdir('./model_tf') :
                    os.mkdir('./model_tf')
                if not os.path.isdir('./record') :
                    os.mkdir('./record')
                k = 0
                while 1 :
                    if os.path.isfile('./model_tf/model_{}_{}.ckpt'.format(self.output_str, k)) :
                        k += 1
                    else :
                        break
                save_path = self.saver.save(self.sess, './model_tf/model_{}_{}.ckpt'.format(self.output_str,k))
                with open('./record/reward_{}_{}.pkl'.format(self.output_str,k), 'wb') as f:
                    pickle.dump(lst_reward_his, f)
                with open('./record/loss_{}_{}.pkl'.format(self.output_str,k), 'wb') as f:
                    pickle.dump(lst_loss_his, f)
                print("Model saved in file: %s" % save_path)

        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        import random
        import numpy as np
        import keras
        import tensorflow as tf
        
        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)
        
        temp = self.prepro(observation)
        state = temp - self.state_prev
        self.state_prev = temp
        
        
        action_p = self.sess.run(self.a_prob, feed_dict={self.input_s: state.reshape(1,80,80,1)}).ravel()
        action = int(np.random.choice(self.action_n,p=action_p))
        if self.action2 :
            if action == 0 :
                action = 2
            elif action == 1 :
                action = 5
        print (action)
        return action
#         return self.env.get_random_action()

