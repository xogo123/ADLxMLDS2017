# initialization ones

### three questions
### reward is negative
### training action is defferent but test action all same
### batch action is same

from agent_dir.agent import Agent

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)
        self.args = args
        
        ##################
        # YOUR CODE HERE #
        ##################


        ### import ###
        import os
        import sys
        import pickle
        import random
        import math
        import numpy as np
        import keras
        from keras import backend as K
        import tensorflow as tf
        
        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)
        
#         def init():
#             config = tf.ConfigProto()
#             config.gpu_options.allow_growth = True
#             session = tf.Session(config=config)
#             keras.backend.tensorflow_backend.set_session(session)
#         init()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        ### parameter ###
#         best_model_name = 'model_GOR_5.ckpt'
        best_model_name = 'model_normal_3.ckpt'
#         best_model_name = 'model_0.05_1.ckpt'
        
        self.bs = self.args.batch_size
        self.lr = self.args.learning_rate
        self.gamma = 0.99
        self.frame_n = 4
        self.iteration = 1e6
        self.iter_update_Q = 35
        self.game_over_reward = 0
        self.max_len_RM = 10000 # max len of replay_memory
        self.ER = 0.05
#         self.life_total = 5
        self.ddqn = False
        self.temp_str = 'doubleDQN'
        self.duelingdqn = False
#         self.temp_str = 'duelingDQN'


        
        if self.game_over_reward :
            self.GOR = '_GOR'
        else :
            self.GOR = ''
            
        self.output_str = '{}_{}_{}'.format(self.GOR, self.ER, self.temp_str)
        
        RM = [] # replay memory [state, action, state_, reward], st is np.array (84,84,4), at is int, rt is float
        action_space = self.env.get_action_space()
        state = self.env.reset()
        
        def build_model() :
            ### model ###
            self.input_s = tf.placeholder(tf.float32, shape=(None,84,84,4))
            self.input_a = tf.placeholder(tf.float32, shape=(None,4))
            self.input_s_ = tf.placeholder(tf.float32, shape=(None,84,84,4))
            self.input_y = tf.placeholder(tf.float32, shape=(None,1))

    #         with tf.variable_scope('Q_temp') :
    #             c = tf.layers.conv2d(input_s, 32, (8,8), strides=(4,4), activation=tf.nn.relu)
    #             c = tf.layers.conv2d(c, 64, (4,4), strides=(2,2), activation=tf.nn.relu)
    #             c = tf.layers.conv2d(c, 64, (3,3), strides=(1,1), activation=tf.nn.relu)
    #             f = tf.reshape(c, [-1, 7*7*64])
    #             d = tf.layers.dense(f, 512, activation=tf.nn.relu)
    #             self.Q_temp = tf.layers.dense(d, 4, activation=None)

    #         with tf.variable_scope('Q_target') :
    #             c = tf.layers.conv2d(input_s, 32, (8,8), strides=(4,4), activation=tf.nn.relu)
    #             c = tf.layers.conv2d(c, 64, (4,4), strides=(2,2), activation=tf.nn.relu)
    #             c = tf.layers.conv2d(c, 64, (3,3), strides=(1,1), activation=tf.nn.relu)
    #             f = tf.reshape(c, [-1, 7*7*64])
    #             d = tf.layers.dense(f, 512, activation=tf.nn.relu)
    #             self.Q_target = tf.layers.dense(d, 4, activation=None)
            with tf.variable_scope('Q_temp') :
                c = keras.layers.Conv2D(32, (8,8), strides=(4,4), activation='relu')(self.input_s)
                c = keras.layers.Conv2D(64, (4,4), strides=(2,2), activation='relu')(c)
                c = keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu')(c)
                f = keras.layers.Flatten()(c)
                d = keras.layers.Dense(512, activation='relu')(f)
                if self.duelingdqn :
                    dueling_temp = keras.layers.Conv2D(32, (8,8), strides=(4,4), activation='relu')(self.input_s)
                    dueling_temp = keras.layers.Flatten()(dueling_temp)
                    dueling_temp = keras.layers.Dense(1, activation='linear')(dueling_temp)
                    d = tf.concat([d, dueling_temp], axis=1)
                self.Q_temp = keras.layers.Dense(4, activation='linear')(d)


            with tf.variable_scope('Q_target') :
                c = keras.layers.Conv2D(32, (8,8), strides=(4,4), activation='relu')(self.input_s_)
                c = keras.layers.Conv2D(64, (4,4), strides=(2,2), activation='relu')(c)
                c = keras.layers.Conv2D(64, (3,3), strides=(1,1), activation='relu')(c)
                f = keras.layers.Flatten()(c)
                d = keras.layers.Dense(512, activation='relu')(f)
                if self.duelingdqn :
                    dueling_target = keras.layers.Conv2D(32, (8,8), strides=(4,4), activation='relu')(self.input_s_)
                    dueling_target = keras.layers.Flatten()(dueling_target)
                    dueling_target = keras.layers.Dense(1, activation='linear')(dueling_target)
                    d = tf.concat([d, dueling_target], axis=1)
                self.Q_target = keras.layers.Dense(4, activation='linear')(d)
                

            lst_Q_target_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q_target')            
            lst_Q_temp_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Q_temp')

            with tf.variable_scope('Q_target_update'):
                self.target_update_op = [tf.assign(t, e) for t, e in zip(lst_Q_target_variable, lst_Q_temp_variable)]

            self.Q_temp_qv_1 = tf.reduce_sum(tf.multiply(self.Q_temp, self.input_a), axis=1, keep_dims=True)
    #         Q_target_qv_1 = tf.reduce_sum(tf.multiply(self.Q_target, input_a), axis=1, keep_dims=True)
    #         Q_target_amax = tf.argmax(Q_target_qv_1, axis=1)[0]

            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.input_y, predictions=self.Q_temp_qv_1))
            self.train_step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            tf.summary.FileWriter("logs/", self.sess.graph)
        
        build_model()

        if args.test_dqn:
            #you can load your model here
            k = 0
            print('loading trained model')
            self.saver.restore(self.sess, './model_tf/{}'.format(best_model_name))
            print ('loading model finished...')



    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        state = self.env.reset()
        
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
        
        random.seed(1)
        np.random.seed(1)
        tf.set_random_seed(1)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
#         def init():
#             config = tf.ConfigProto()
#             config.gpu_options.allow_growth = True
#             session_keras = tf.Session(config=config)
#             keras.backend.tensorflow_backend.set_session(session_keras)
#         init()
        
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
        
        ### parameter ###
        bs = self.bs
        lr = self.lr
        gamma = self.gamma
        frame_n = self.frame_n
        iteration = self.iteration
        iter_update_Q = self.iter_update_Q
        max_len_RM = self.max_len_RM # max len of replay_memory
#         life_total = self.life_total
        RM = [] # replay memory [state, action, state_, reward], st is np.array (84,84,4), at is int, rt is float
        action_space = self.env.get_action_space()
        state = self.env.reset()
        
#         def Q(state, action, model) :
#             feed_dict = {state: state, action: action}
            
#             tf.run
        
        ### initialize RM ###
        print ('initialzing replay memory')
        done_n = 0
        for _ in range(max_len_RM) :
#             print (_)
            action = self.env.get_random_action()
            state_, reward, done, info = self.env.step(action)
            if done :
#                 print ('--- game over ---')
                reward = self.game_over_reward
                state_ = self.env.reset()
            RM += [[state, action, state_, reward]]
            state = state_
        
        ### training ###
        ave_reward_30 = 0
        episode_i = 0
        reward_episode = 0
        lst_reward_episode = []
        lst_reward_his = []
        lst_loss_his = []
        for i0 in range(int(iteration)) :
            print (i0)
            ### b for batch ###
            lst_b_state = []
            lst_b_action = []
            lst_b_state_ = []
            lst_b_reward = []
            lst_b_y = []
            
            ### random pick batch from replay memory ###
            random.shuffle(RM)
            for _ in range(self.bs) :
                sample = RM[_]
                lst_b_state += [sample[0]]
                lst_b_action += [sample[1]]
                lst_b_state_ += [sample[2]]
                lst_b_reward += [sample[3]]
            RM = RM[32:]
            
            if self.ddqn :
                lst_ttemp = []
                Q_target_max_4 = self.sess.run(self.Q_target, feed_dict={self.input_s_: np.asarray(lst_b_state)}) # shape=(32,4)
                Q_target_max = np.argmax(Q_target_max_4, axis=1)
                Q_target_max_4 = self.sess.run(self.Q_target, feed_dict={self.input_s_: np.asarray(lst_b_state_)}) # shape=(32,4)
                for i_tmp,temp_max in enumerate(Q_target_max) :
                    lst_ttemp += [Q_target_max_4[i_tmp,temp_max].tolist()]
                Q_target_max = np.asarray(lst_ttemp)
                ary_b_y = np.asarray(lst_b_reward) + (gamma * Q_target_max) # shpae=(32,)
            else :
                Q_target_max_4 = self.sess.run(self.Q_target, feed_dict={self.input_s_: np.asarray(lst_b_state_)}) # shape=(32,4)
                Q_target_max = np.max(Q_target_max_4, axis=1)
                ary_b_y = np.asarray(lst_b_reward) + (gamma * Q_target_max) # shpae=(32,)
                
            ## add new one to RM ###
            for _ in range(self.bs) :
                if i0 > 35000 :
                    EE_threshold = self.ER
                else :
                    EE_threshold = max(1 - (0.95 * i0 / 35000.), self.ER) # exploration and exploitation
                if random.random() < EE_threshold :
                    action = self.env.get_random_action()
    #                     print ('++++++++++random action+++++++++++')
                else :
    #                     print ('### best action ###')
                    action = np.argmax(self.sess.run(self.Q_target, feed_dict={self.input_s_: np.asarray(state).reshape(1,84,84,4)}))
#                     print (action)

                state_, reward, done, info = self.env.step(action)
                if done :
                    reward = self.game_over_reward
                    reward_episode += reward
                    state_ = self.env.reset()
                    lst_reward_episode += [reward_episode]
                    reward_episode = 0
                    episode_i += 1
                    print ('episode_i : {}'.format(episode_i))
                    print ('ave_reward_30 : {}'.format(np.mean(lst_reward_episode[-30:])))
                    print ('\n')
                elif reward > 0 :
                    reward_episode += reward
                    print ('++++ reward : {} ++++'.format(reward_episode))
                
                RM += [[state, action, state_, reward]]
                state = state_
                    
            action_one_hot = keras.utils.to_categorical(lst_b_action, num_classes=4).reshape(-1,4)
            
            _, loss_temp = self.sess.run([self.train_step,self.loss], feed_dict={self.input_s: np.asarray(lst_b_state),
                                                                   self.input_a: action_one_hot,
                                                                   self.input_y: ary_b_y.reshape(-1,1)})
            
            lst_loss_his += [loss_temp]
            lst_reward_his += [np.mean(lst_reward_episode[-30:])]
#             Q_temp.train_on_batch([np.asarray(lst_b_state), action_one_hot], np.asarray(lst_b_y))
#             Q_temp.fit([np.asarray(lst_b_state), keras.utils.to_categorical(lst_b_action, num_classes=4).reshape(-1,4)], np.asarray(lst_b_y), epochs=1, batch_size=bs, validation_split=0., callbacks=[])

            ### update Q_target ###
            if i0 % iter_update_Q == 0 :
                print ('---- updating Q_target ----')
                self.sess.run(self.target_update_op)

            ## saving model ###
            if i0 % 3000 == 2999 :
                print ('saving model...')
                if not os.path.isdir('./model_tf') :
                    os.mkdir('./model_tf')
                if not os.path.isdir('./record') :
                    os.mkdir('./record')
                k = 0
                while 1 :
                    if os.path.isfile('./model_tf/{}_{}.ckpt'.format(self.output_str, k)) :
                        k += 1
                    else :
                        break
                save_path = self.saver.save(self.sess, './model_tf/model_{}_{}.ckpt'.format(self.output_str, k))
                with open('./record/reward_{}_{}.pkl'.format(self.output_str, k), 'wb') as f:
                    pickle.dump(lst_reward_his, f)
                with open('./record/loss_{}_{}.pkl'.format(self.output_str, k), 'wb') as f:
                    pickle.dump(lst_loss_his, f)
                print("Model saved in file: %s" % save_path)
        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

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
        
        
        if random.random() < 0.01 :
            action = self.env.get_random_action()
        else :
            Q_target_max_4 = self.sess.run(self.Q_target, feed_dict={self.input_s_: np.asarray(observation).reshape(1,84,84,4)})
            action = np.argmax(np.asarray(Q_target_max_4))
        
#         print (action)
        return action
#         return self.env.get_random_action()

