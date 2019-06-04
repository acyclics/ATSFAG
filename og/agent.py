import numpy as np 
import tensorflow as tf 
import random

class Episode_experience():
    def __init__(self):
        self.memory = []
        
    def add(self, state, action, reward, next_state, done, goal):
        self.memory += [(state, action, reward, next_state, done, goal)]
        
    def clear(self):
        self.memory = []

class DDPGAgent:
    def __init__(self, state_size, action_size, goal_size, action_low=-1, action_high=1, gamma=0.98,
                 actor_learning_rate=0.01, critic_learning_rate=0.01, tau=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.goal_size = goal_size
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma   # discount rate
        self.memory = []
        self.buffer_size = int(5e4)
        self.a_learning_rate = actor_learning_rate
        self.c_learning_rate = critic_learning_rate # often larger than actor_learning_rate
        self.tau = tau # soft update
        self.batch_size = 32
        self.gradient_norm_clip = None
        self._construct_nets()
        
    def _construct_nets(self):
        tf.reset_default_graph()
        self.sess = tf.Session()
        
        self.S = tf.placeholder(tf.float32, [None, self.state_size], 'state')
        self.S_ = tf.placeholder(tf.float32, [None, self.state_size], 'next_state')
        self.G = tf.placeholder(tf.float32, [None, self.goal_size], 'goal')
        self.D = tf.placeholder(tf.float32, [None, ], 'done')
        self.R = tf.placeholder(tf.float32, [None, ], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, self.G, scope='eval')
            self.a_ = self._build_a(self.S_, self.G, scope='target')
        with tf.variable_scope('Critic'):
            self.q = self._build_c(self.S, self.a, self.G, scope='eval')
            self.q_ = self._build_c(self.S_, self.a_, self.G, scope='target')
        
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.soft_update_op = [[tf.assign(ta, (1 - self.tau) * ta + self.tau * ea), tf.assign(tc, (1 - self.tau) * tc + self.tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + self.gamma * (1-self.D) * self.q_
        
        self.c_loss = tf.losses.mean_squared_error(q_target, self.q)
        self.a_loss = - tf.reduce_mean(self.q)    # maximize the q
        
        if self.gradient_norm_clip is not None:
            c_optimizer = tf.train.AdamOptimizer(self.c_learning_rate)
            c_gradients = c_optimizer.compute_gradients(self.c_loss, var_list=self.ce_params)
            for i, (grad, var) in enumerate(c_gradients):
                if grad is not None:
                    c_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm_clip), var)
            self.c_train = c_optimizer.apply_gradients(c_gradients)
            a_optimizer = tf.train.AdamOptimizer(self.a_learning_rate)
            a_gradients = c_optimizer.compute_gradients(self.a_loss, var_list=self.ae_params)
            for i, (grad, var) in enumerate(a_gradients):
                if grad is not None:
                    a_gradients[i] = (tf.clip_by_norm(grad, self.gradient_norm_clip), var)
            self.a_train = a_optimizer.apply_gradients(a_gradients)
        else:
            self.c_train = tf.train.AdamOptimizer(self.c_learning_rate).minimize(self.c_loss, var_list=self.ce_params)
            self.a_train = tf.train.AdamOptimizer(self.a_learning_rate).minimize(self.a_loss, var_list=self.ae_params)
            
        self.saver = tf.train.Saver()
        
        self.sess.run(tf.global_variables_initializer())
    
    def _build_a(self, s, g, scope): # policy
        with tf.variable_scope(scope):
            net = tf.concat([s, g], 1)
            net = tf.layers.dense(net, 64, tf.nn.relu)
            net = tf.layers.dense(net, 64, tf.nn.relu)
            a = tf.layers.dense(net, self.action_size, tf.nn.tanh)
            return a * (self.action_high-self.action_low)/2 + (self.action_high+self.action_low)/2
    
    def _build_c(self, s, a, g, scope): # Q value
        with tf.variable_scope(scope):
            net = tf.concat([s, a, g], 1)
            net = tf.layers.dense(net, 64, tf.nn.relu)
            net = tf.layers.dense(net, 64, tf.nn.relu)
            return tf.layers.dense(net, 1)
    
    def choose_action(self, state, goal, variance): # normal distribution
        action = self.sess.run(self.a, {self.S: state, self.G: goal})[0]
        return np.clip(np.asarray([np.random.normal(action[0], variance), np.random.normal(action[1], (variance/6.18) * 0.785)]), self.action_low, self.action_high)
    
    def remember(self, ep_experience):
        self.memory += ep_experience.memory
        if len(self.memory) > self.buffer_size:
            self.memory = self.memory[-self.buffer_size:] # empty the first memories
        
    def replay(self, optimization_steps=1):
        if len(self.memory) < self.batch_size: # if there's no enough transitions, do nothing
            return 0, 0
        
        a_losses = 0
        c_losses = 0
        for _ in range(optimization_steps):
            minibatch = np.vstack(random.sample(self.memory, self.batch_size))
            ss = np.vstack(minibatch[:,0])
            acs = np.vstack(minibatch[:,1])
            rs = minibatch[:,2]
            nss = np.vstack(minibatch[:,3])
            ds = minibatch[:,4]
            gs = np.vstack(minibatch[:,5])
            a_loss, c_loss, _, _ = self.sess.run([self.a_loss, self.c_loss, self.a_train, self.c_train],
                                                 {self.S: ss, self.a: acs, self.R: rs,
                                                  self.S_: nss, self.D: ds, self.G: gs})
            a_losses += a_loss
            c_losses += c_loss
            
        return a_losses/optimization_steps, c_losses/optimization_steps
    
    def update_target_net(self):
        self.sess.run(self.soft_update_op)
