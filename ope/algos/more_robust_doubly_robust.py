import sys
import numpy as np
import pandas as pd
from copy import deepcopy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras
from keras.models     import Sequential
from keras.layers     import Dense, Conv2D, Flatten, MaxPool2D, concatenate, UpSampling2D, Reshape, Lambda
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from tqdm import tqdm
from ope.utls.thread_safe import threadsafe_generator
from keras import regularizers
from sklearn.linear_model import LinearRegression, LogisticRegression

class MRDR(object):
    def __init__(self, data, gamma, frameskip=2, frameheight=2, modeltype = 'conv', processor=None):
        self.data = data
        self.gamma = gamma
        self.frameskip = frameskip
        self.frameheight = frameheight
        self.modeltype = modeltype
        self.processor = processor

    def q_beta(self):
        return tf.matmul(tf.matrix_diag(self.pi1),  tf.expand_dims(self.out,2)) - tf.expand_dims(self.rew, 2)

    def Q_val(self):
        with tf.variable_scope('w', reuse = tf.AUTO_REUSE):

            # h = tf.layers.conv2d(self.s, 32, 4, data_format='channels_first', strides=2, activation=tf.nn.relu, name="enc_conv1")
            # h = tf.layers.conv2d(h, 64, 4, data_format='channels_first', strides=2, activation=tf.nn.relu, name="enc_conv2")
            # h = tf.layers.conv2d(h, 128, 4, data_format='channels_first', strides=2, activation=tf.nn.relu, name="enc_conv3")
            # h = tf.layers.conv2d(h, 256, 4, data_format='channels_first', strides=2, activation=tf.nn.relu, name="enc_conv4")
            # h = tf.reshape(h, [-1, 3*5*256])

            # # Some dense layers
            s = tf.layers.flatten(self.s)
            dense1 = tf.layers.dense(s, 16, activation=tf.nn.relu, name="dense1", kernel_regularizer = tf.contrib.layers.l2_regularizer(1.), bias_regularizer = tf.contrib.layers.l2_regularizer(1.))
            dense2 = tf.layers.dense(dense1, 8, activation=tf.nn.relu, name="dense2", kernel_regularizer = tf.contrib.layers.l2_regularizer(1.), bias_regularizer = tf.contrib.layers.l2_regularizer(1.))
            out = tf.layers.dense(dense2, self.action_dim, name="Q")

            return out

    def build_model_(self, input_size, scope, action_space_dim=3, modeltype='conv'):
        # place holder
        self.action_dim = action_space_dim
        # tio2 = tf.placeholder(tf.float32, [None], name='policy_ratio2')


        self.s = tf.placeholder(tf.float32, [None] + list(input_size), name='state')
        # self.gam = tf.placeholder(tf.float32, [None] , name='gamma_sq')
        # self.omega_cumul = tf.placeholder(tf.float32, [None] , name='cumulative_omega')
        # self.omega = tf.placeholder(tf.float32, [None] , name='current_omega')
        self.factor = tf.placeholder(tf.float32, [None] , name='current_omega')
        self.rew = tf.placeholder(tf.float32, [None] + [self.action_dim], name='disc_future_rew')
        self.pi0 = tf.placeholder(tf.float32, [None] + [self.action_dim], name='pi_b')
        self.pi1 = tf.placeholder(tf.float32, [None] + [self.action_dim], name='pi_e')


        # self.factor = self.gam * (self.omega_cumul**2) * self.omega
        self.Omega = tf.matrix_diag(1/self.pi0)
        self.Omega = self.Omega - tf.ones_like(self.Omega)

        self.out = self.Q_val()
        self.q = self.q_beta()

        self.loss_pre_reduce = tf.matmul(tf.matmul(tf.transpose(self.q, [0,2,1]) , self.Omega), self.q)
        self.loss_pre_reduce_w_factor = self.factor * tf.squeeze(self.loss_pre_reduce)
        self.loss = tf.reduce_sum(self.loss_pre_reduce_w_factor)

        self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'mrdr'))
        LR = .001
        reg_weight = 0.
        self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss + reg_weight * self.reg_loss)

        # initialize vars
        self.init = tf.global_variables_initializer()

        # Create assign opsfor VAE
        t_vars = tf.trainable_variables()
        self.assign_ops = {}
        for var in t_vars:
            if var.name.startswith('mrdr'):
                pshape = var.get_shape()
                pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')
                assign_op = var.assign(pl)
                self.assign_ops[var] = (assign_op, pl)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def build_model(self, input_size, scope, action_space_dim=3, modeltype='conv'):
        inp = keras.layers.Input(input_size, name='frames')
        actions = keras.layers.Input((action_space_dim,), name='mask')
        weights = keras.layers.Input((1,), name='weights')
        rew = keras.layers.Input((action_space_dim,), name='rewards')
        pib = keras.layers.Input((action_space_dim,), name='pi_b')
        pie = keras.layers.Input((action_space_dim,), name='pi_e')


        def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=np.random.randint(2**32))
        if modeltype == 'conv':
            conv1 = Conv2D(8, (7,7), strides=(3,3), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(inp)
            pool1 = MaxPool2D(data_format='channels_first')(conv1)
            conv2 = Conv2D(16, (3,3), strides=(1,1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(pool1)
            pool2 = MaxPool2D(data_format='channels_first')(conv2)
            flat1 = Flatten(name='flattened')(pool2)
            out = Dense(256, activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(flat1)
        elif modeltype == 'conv1':
            def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001, seed=np.random.randint(2**32))
            conv1 = Conv2D(16, (2,2), strides=(1,1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(inp)
            # pool1 = MaxPool2D(data_format='channels_first')(conv1)
            # conv2 = Conv2D(16, (2,2), strides=(1,1), padding='same', data_format='channels_first', activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(pool1)
            # pool2 = MaxPool2D(data_format='channels_first')(conv2)
            flat1 = Flatten(name='flattened')(conv1)
            out = Dense(8, activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(flat1)
            out = Dense(8, activation='elu',kernel_initializer=init(), bias_initializer=init(), kernel_regularizer=regularizers.l2(1e-6))(out)
        else:
            def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=.001, seed=np.random.randint(2**32))
            flat = Flatten()(inp)
            dense1 = Dense(16, activation='relu',kernel_initializer=init(), bias_initializer=init())(flat)
            # dense2 = Dense(256, activation='relu',kernel_initializer=init(), bias_initializer=init())(dense1)
            dense3 = Dense(8, activation='relu',kernel_initializer=init(), bias_initializer=init())(dense1)
            out = Dense(4, activation='relu', name='out',kernel_initializer=init(), bias_initializer=init())(dense3)


        all_actions = Dense(action_space_dim, name=scope + 'all_Q', activation="linear",kernel_initializer=init(), bias_initializer=init())(out)

        output = keras.layers.dot([all_actions, actions], 1)

        model = keras.models.Model(inputs=[inp, actions, weights, rew, pib, pie], outputs=[all_actions])

        all_Q = keras.models.Model(inputs=[inp],
                                 outputs=model.get_layer(scope + 'all_Q').output)

        rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.95, epsilon=1e-08, decay=1e-3)#, clipnorm=1.)
        # adam = keras.optimizers.Adam()

        model.add_loss(self.MRDR_loss(all_actions, weights, rew, pib, pie))
        model.compile(loss=None, optimizer=rmsprop, metrics=['accuracy'])

        def get_gradient_norm(model):
            with K.name_scope('gradient_norm'):
                grads = K.gradients(model.total_loss, model.trainable_weights)
                norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            return norm

        # Append the "l2 norm of gradients" tensor as a metric
        model.metrics_names.append("gradient_norm")
        model.metrics_tensors.append(get_gradient_norm(model))

        return model, all_Q

    def MRDR_loss(self, Q, weights, rew, pib, pie):

        # There is numerical instability here on MacOSX. Loss goes negative in TF when overfitting to 1 datapoint but stays positive in Numpy
        # sess = K.get_session()
        # Omega = sess.run(self.Omega, feed_dict={self.Q_k.input[0]: x,self.Q_k.input[1]: acts,self.Q_k.input[3]:rs, self.Q_k.input[4]:pib, self.Q_k.input[5]:pie})
        # sess.run(self.D, feed_dict={self.Q_k.input[0]: x,self.Q_k.input[1]: acts,self.Q_k.input[3]:rs, self.Q_k.input[4]:pib, self.Q_k.input[5]:pie})
        # sess.run(self.Q, feed_dict={self.Q_k.input[0]: x,self.Q_k.input[1]: acts,self.Q_k.input[3]:rs, self.Q_k.input[4]:pib, self.Q_k.input[5]:pie})
        # qbeta= sess.run(self.qbeta, feed_dict={self.Q_k.input[0]: x,self.Q_k.input[1]: acts,self.Q_k.input[3]:rs, self.Q_k.input[4]:pib, self.Q_k.input[5]:pie})
        # sess.run(self.unweighted_loss, feed_dict={self.Q_k.input[0]: x,self.Q_k.input[1]: acts,self.Q_k.input[3]:rs, self.Q_k.input[4]:pib, self.Q_k.input[5]:pie})
        # sess.run(self.loss, feed_dict={self.Q_k.input[0]: x,self.Q_k.input[1]: acts,self.Q_k.input[3]:rs, self.Q_k.input[4]:pib, self.Q_k.input[5]:pie})


        # Omega = np.array([np.diag(1/x) for x in pib])
        # Omega -= 1 #tf.ones_like(Omega)
        # D = np.array([np.diag(x) for x in pie])
        # qbeta = np.matmul(D, np.expand_dims(self.Q_k_all.predict(x),2)) - np.expand_dims(rs, 2)
        # qbeta_T = np.transpose(qbeta, [0,2,1])
        # unweighted_loss = np.matmul(np.matmul(qbeta_T, Omega), qbeta)
        # loss = np.reshape(weights, (-1,1)) * np.reshape(unweighted_loss, (-1,1))
        # return np.reduce_mean(loss)
        self.Q = Q
        self.rew =rew
        self.Omega = tf.matrix_diag(tf.math.divide(1,pib))
        self.Omega -= 1 #tf.ones_like(Omega)
        self.D = tf.matrix_diag(pie)
        self.qbeta = tf.matmul(self.D, tf.expand_dims(Q,2)) - tf.expand_dims(self.rew, 2)
        qbeta_T = tf.transpose(self.qbeta, [0,2,1])
        self.unweighted_loss = tf.matmul(tf.matmul(qbeta_T, self.Omega), self.qbeta)
        self.loss = tf.reduce_mean(self.unweighted_loss)
        self.weighted_loss = tf.reshape(weights, (-1,1)) * tf.reshape(self.unweighted_loss, (-1,1)) #weights * self.unweighted_loss
        return tf.reduce_mean(tf.squeeze(self.weighted_loss))

    def run_NN_tf(self, env, pi_b, pi_e, max_epochs, epsilon=0.001):

        initial_states = self.data.initial_states()
        self.dim_of_actions = env.n_actions
        self.Q_k = None

        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=1e-4,  patience=10, verbose=1, mode='min', restore_best_weights=True)
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')

        self.more_callbacks = [earlyStopping, reduce_lr_loss]
        im = self.data.states()[0]
        self.g = tf.Graph()
        with self.g.as_default():
            with tf.variable_scope('mrdr', reuse=False):
                self.build_model_(im.shape[1:], 'Q_k', modeltype=self.modeltype, action_space_dim=env.n_actions)
                self._init_session()
        # self.build_model(im.shape[1:], 'Q_k', modeltype=self.modeltype, action_space_dim=env.n_actions)

        batch_size = 32

        dataset_length = self.data.num_tuples()
        perm = np.random.permutation(range(dataset_length))

        perm = np.random.permutation(self.data.idxs_of_non_abs_state())

        eighty_percent_of_set = int(1.*len(perm))
        training_idxs = perm[:eighty_percent_of_set]
        validation_idxs = perm[eighty_percent_of_set:]
        training_steps_per_epoch = int(1. * np.ceil(len(training_idxs)/float(batch_size)))
        validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batch_size)))
        # steps_per_epoch = 1 #int(np.ceil(len(dataset)/float(batch_size)))
        train_gen = self.generator(env, pi_e, training_idxs, fixed_permutation=True, batch_size=batch_size)
        val_gen = self.generator(env, pi_e, validation_idxs, fixed_permutation=True, batch_size=batch_size, is_train=False)

        for i in range(1):
            for j in range(max_epochs):
                totloss = 0
                for k in range(training_steps_per_epoch):
                    x,y = next(train_gen)
                    # import pdb; pdb.set_trace()
                    _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
                        self.s: x[0],
                        self.factor: x[2],
                        self.rew: x[3],
                        self.pi0: x[4],
                        self.pi1: x[5],
                        })
                    totloss += loss*x[0].shape[0]
                print(i, j, k, totloss)


        return [],[],self

    def run_NN(self, env, pi_b, pi_e, max_epochs, batch_size, epsilon=0.001):

        initial_states = self.data.initial_states()
        self.dim_of_actions = env.n_actions
        self.Q_k, self.Q_k_all = None, None

        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=1e-4,  patience=10, verbose=1, mode='min', restore_best_weights=True)
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')

        self.more_callbacks = [earlyStopping, reduce_lr_loss]

        im = self.data.states()[0]
        if self.processor: im = self.processor(im)
        self.Q_k, self.Q_k_all = self.build_model(im.shape[1:], 'Q_k', modeltype=self.modeltype, action_space_dim=env.n_actions)
        # self.Q_k_minus_1, self.Q_k_minus_1_all = self.build_model(im.shape[1:], 'Q_k_minus_1', modeltype=self.modeltype)
        values = []
        self.Q_k_all.predict([[im[0]]])

        print('Training: MRDR')
        losses = []
        for k in tqdm(range(1)):

            dataset_length = self.data.num_tuples()
            perm = np.random.permutation(range(dataset_length))

            perm = np.random.permutation(self.data.idxs_of_non_abs_state())

            eighty_percent_of_set = int(.8*len(perm))
            training_idxs = perm[:eighty_percent_of_set]
            validation_idxs = perm[eighty_percent_of_set:]
            training_steps_per_epoch = int(1. * np.ceil(len(training_idxs)/float(batch_size)))
            validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batch_size)))
            # steps_per_epoch = 1 #int(np.ceil(len(dataset)/float(batch_size)))
            train_gen = self.generator(env, pi_e, training_idxs, fixed_permutation=True, batch_size=batch_size)
            val_gen = self.generator(env, pi_e, validation_idxs, fixed_permutation=True, batch_size=batch_size, is_train=False)

            # import pdb; pdb.set_trace()
            # train_gen = self.generator(env, pi_e, (transitions,frames), training_idxs, fixed_permutation=True, batch_size=batch_size)
            # inp, out = next(train_gen)
            M = 5
            hist = self.Q_k.fit_generator(train_gen,
                               steps_per_epoch=training_steps_per_epoch,
                               # validation_data=val_gen,
                               # validation_steps=validation_steps_per_epoch,
                               epochs=max_epochs,
                               max_queue_size=50,
                               workers=1,
                               use_multiprocessing=False,
                               verbose=1,
                               callbacks = self.more_callbacks)


        return np.mean(values[-10:]), self.Q_k, self.Q_k_all

    @threadsafe_generator
    def generator(self, env, pi_e, all_idxs, fixed_permutation=False,  batch_size = 64, is_train=True):
        # dataset, frames = dataset
        data_length = len(all_idxs)
        steps = int(np.ceil(data_length/float(batch_size)))

        n = len(self.data)
        T = max(self.data.lengths())
        n_dim = self.data.n_dim
        n_actions = self.data.n_actions

        data_dim = n * n_actions * T
        omega = n*T * [None]
        propensity_weights = []
        # r_tild = np.zeros(data_dim)
        # for i in tqdm(range(n)):


        #     states = np.squeeze(self.data.states(low_=i, high_=i+1))
        #     actions = self.data.actions()[i]
        #     rewards = self.data.rewards()[i]
        #     pi0 = self.data.base_propensity()[i]
        #     pi1 = self.data.target_propensity()[i]
        #     l = self.data.lengths()[i]

        #     for t in range(min(T, l)):
        #         state_t = states[t]
        #         action_t = actions[t]
        #         reward_t = rewards[t]
        #         pib_s_t = pi0[t] #self.policy_behave.pi(state_t)
        #         pie_s_t = pi1[t] #self.policy_eval.pi(state_t)
        #         omega_s_t = np.diag(1 / pib_s_t) - 1 #np.ones((n_actions, n_actions))
        #         D_pi_e = np.diag(pie_s_t)
        #         if t == 0:
        #             rho_prev = 1
        #         else:
        #             rho_prev =  self.rho[i][t-1]

        #         propensity_weight_t = gamma_vec[t] ** 2 * rho_prev ** 2 * (self.rho[i][t] / rho_prev)

        #         propensity_weights.append(propensity_weight_t)

        #         om = propensity_weight_t * D_pi_e.dot(omega_s_t).dot(D_pi_e)
        #         omega[i * T + t] = om

        #         t_limit = min(T, l)
        #         r_tild[(i * T + t) * n_actions + action_t] = np.sum((self.rho[i][t:t_limit] / self.rho[i][t]) *
        #                                (gamma_vec[t:t_limit] / gamma_vec[t]) * rewards[t:])

        # Rs = np.array(r_tild).reshape(-1, env.n_actions)
        omega = [np.cumprod(om) for om in self.data.omega()]
        gamma_vec = self.gamma**np.arange(T)
        actions = self.data.actions()
        rewards = self.data.rewards()

        factors, Rs = [], []
        for traj_num, ts in tqdm(enumerate(self.data.ts())):
            for t in ts:
                i,t = int(traj_num), int(t)
                R = np.zeros(env.n_actions)
                if omega[i][t]:
                    R[actions[i,t]] = np.sum( omega[i][t:]/omega[i][t] * gamma_vec[t:]/gamma_vec[t] *  rewards[i][t:] )
                else:
                    R[actions[i,t]] = 0

                Rs.append(R)

                if t == 0:
                    rho_prev = 1
                else:
                    rho_prev =  omega[i][t-1]

                if rho_prev:
                    propensity_weight_t = gamma_vec[t] ** 2 * rho_prev ** 2 * (omega[i][t] / rho_prev)
                else:
                    propensity_weight_t = 0

                factors.append(propensity_weight_t)


        Rs = np.array(Rs)
        factors = np.array(factors) #np.atleast_2d(np.array(factors)).T

        states = self.data.states()
        original_shape = states.shape
        states = states.reshape(-1,np.prod(states.shape[2:]))

        actions = np.eye(env.n_actions)[actions.reshape(-1)]

        base_propensity = self.data.base_propensity().reshape(-1, env.n_actions)
        target_propensity = self.data.target_propensity().reshape(-1, env.n_actions)

        while True:

            perm = np.random.permutation(all_idxs)
            for batch in np.arange(steps):
                batch_idxs = perm[(batch*batch_size):((batch+1)*batch_size)]

                x = states[batch_idxs].reshape(tuple([-1]) + original_shape[2:])
                acts = actions[batch_idxs]

                rs = Rs[batch_idxs]
                weights = factors[batch_idxs] #* probs[batch_idxs] / np.min(probs)
                pib = base_propensity[batch_idxs]
                pie = target_propensity[batch_idxs]

                if self.processor: x = self.processor(x)

                yield ([x, acts, weights, rs, pib, pie], [])

    def run(self, pi_e):

        n = len(self.data)
        T = max(self.data.lengths())
        n_dim = self.data.n_dim
        n_actions = self.data.n_actions
        self.rho = [np.cumprod(om) for om in self.data.omega()]
        gamma_vec = self.gamma**np.arange(T)

        data_dim = n * n_actions * T
        omega = n*T * [None]
        r_tild = np.zeros(data_dim)
        for i in tqdm(range(n)):


            states = np.squeeze(self.data.states(low_=i, high_=i+1))
            actions = self.data.actions()[i]
            rewards = self.data.rewards()[i]
            pi0 = self.data.base_propensity()[i]
            pi1 = self.data.target_propensity()[i]
            l = self.data.lengths()[i]



            for t in range(min(T, l)):
                state_t = states[t]
                action_t = actions[t]
                reward_t = rewards[t]
                pib_s_t = pi0[t] #self.policy_behave.pi(state_t)
                pie_s_t = pi1[t] #self.policy_eval.pi(state_t)
                omega_s_t = np.diag(1 / pib_s_t) - 1 #np.ones((n_actions, n_actions))
                D_pi_e = np.diag(pie_s_t)
                if t == 0:
                    rho_prev = 1
                else:
                    rho_prev =  self.rho[i][t-1]

                if rho_prev:
                    propensity_weight_t = gamma_vec[t] ** 2 * rho_prev ** 2 * (self.rho[i][t] / rho_prev)
                else:
                    propensity_weight_t = 0

                om = propensity_weight_t * D_pi_e.dot(omega_s_t).dot(D_pi_e)
                omega[i * T + t] = om

                t_limit = min(T, l)
                if self.rho[i][t]:
                    val = np.sum((self.rho[i][t:t_limit] / self.rho[i][t]) * (gamma_vec[t:t_limit] / gamma_vec[t]) * rewards[t:])
                else:
                    val = 0
                r_tild[(i * T + t) * n_actions + action_t] = val

        self.alpha = 1
        self.lamb = 1
        self.cond_number_threshold_A = 10000
        block_size = int(n_actions * n * T/4)

        phi = self.compute_grid_features(pi_e)
        self.weights = self.wls_sherman_morrison(phi, r_tild, omega, self.lamb, self.alpha, self.cond_number_threshold_A, block_size)
        return self


    def compute_feature_without_time(self, state, action, step):
        T = max(self.data.lengths())
        n_dim = self.data.n_dim
        n_actions = self.data.n_actions

        if self.modeltype == 'tabular':
            phi = np.zeros((n_dim, n_actions))
            phi[int(state), int(action)] = 1
            phi = phi.reshape(-1)
        elif self.modeltype == 'linear':
            phi = state.reshape(-1)
        else:
            raise

        return phi

    def compute_feature(self, state, action, step):
        return self.compute_feature_without_time(state, action, step)


    def compute_grid_features(self, pi_e):

        n = len(self.data)
        T = max(self.data.lengths())
        n_dim = self.data.n_dim
        n_actions = self.data.n_actions

        data_dim = n * T * n_actions

        phi = data_dim * [None]
        target_propensity = self.data.target_propensity()

        for i in range(n):
            states = np.squeeze(self.data.states(low_=i, high_=i+1))
            l = self.data.lengths()[i]
            for t in range(T):
                for action in range(n_actions):
                    if t < l:
                        s = states[t]
                        pie_s_t_a_t = target_propensity[i][t][action] #pi_e.predict([states[t]])[0][action]
                        phi[(i * T + t) * n_actions + action] = pie_s_t_a_t * self.compute_feature(s, action, t)
                    else:
                        phi[(i * T + t) * n_actions + action] = np.zeros(len(phi[0]))


        return np.array(phi, dtype='float')

    def wls_sherman_morrison(self, phi_in, rewards_in, omega_in, lamb, omega_regularizer, cond_number_threshold_A, block_size=None):
        # omega_in_2 = block_diag(*omega_in)
        # omega_in_2 += omega_regularizer * np.eye(len(omega_in_2))
        # Aw = phi_in.T.dot(omega_in_2).dot(phi_in)
        # Aw = Aw + lamb * np.eye(phi_in.shape[1])
        # print(np.linalg.cond(Aw))
        # bw = phi_in.T.dot(omega_in_2).dot(rewards_in)
        feat_dim = phi_in.shape[1]
        b = np.zeros((feat_dim, 1))
        B = np.eye(feat_dim)
        data_count = len(omega_in)
        if np.isscalar(omega_in[0]):
            omega_size = 1
            I_a = 1
        else:
            omega_size = omega_in[0].shape[0]
            I_a = np.eye(omega_size)

        for i in range(data_count):
            if omega_in[i] is None:
            # if omega_in[i] is None or (omega_size==1 and omega_in[i] == 0):
                #omega_in[i] = I_a
                #rewards_in[i] = 1
                continue
            omeg_i = omega_in[i] + omega_regularizer * I_a
            #if omega_size > 1:
            #    omeg_i = omeg_i / np.max(omeg_i)

            feat = phi_in[i * omega_size: (i + 1) * omega_size, :]
            # A = A + feat.T.dot(omega_list[i]).dot(feat)
            rews_i = np.reshape(rewards_in[i * omega_size: (i + 1) * omega_size], [omega_size, 1])

            b = b + feat.T.dot(omeg_i).dot(rews_i)

            #  Sherman–Morrison–Woodbury formula:
            # (B + UCV)^-1 = B^-1 - B^-1 U ( C^-1 + V B^-1 U)^-1 V B^-1
            # in our case: U = feat.T   C = omega_list[i]  V = feat
            # print(omeg_i)
            if omega_size > 1:
                C_inv = np.linalg.inv(omeg_i)
            else:
                C_inv = 1/omeg_i
            if np.linalg.norm(feat.dot(B).dot(feat.T)) < 0.0000001:
                inner_inv = omeg_i
            else:
                inner_inv = np.linalg.inv(C_inv + feat.dot(B).dot(feat.T))

            B = B - B.dot(feat.T).dot(inner_inv).dot(feat).dot(B)

        weight_prim = B.dot(b)
        weight = weight_prim.reshape((-1,))
        return weight

    def predict(self, x):
        if (self.data.n_dim + self.data.n_actions) == x.shape[1]:
            acts = np.argmax(x[:,-self.data.n_actions:], axis=1)
            S = x[:,:self.data.n_dim]

            Q = np.zeros(x.shape[0])
            for i, (s, a) in enumerate(zip(S, acts)):
                s = int(s)
                a = int(a)
                Q[i] = np.matmul(self.weights, self.compute_feature(s, a, 0))

            return Q
        elif (1 + self.data.n_actions) == x.shape[1]:
            acts = np.argmax(x[:,-self.data.n_actions:], axis=1)
            S = x[:,:1]

            Q = np.zeros(x.shape[0])
            for i, (s, a) in enumerate(zip(S, acts)):
                Q[i] = np.matmul(self.weights, self.compute_feature(s, a, 0))

            return Q
        elif self.modeltype == 'linear':
            acts = np.argmax(x[:,-self.data.n_actions:], axis=1)
            S = x[:,:-self.data.n_actions]
            Q = np.zeros(x.shape[0])
            for i, (s, a) in enumerate(zip(S, acts)):
                Q[i] = np.matmul(self.weights, self.compute_feature(s, a, 0))

            return Q
        else:
            out = self.sess.run([self.out], feed_dict={
                            self.s: x
                            })
            return out[0]














# import numpy as np
# import tensorflow as tf
# from time import sleep
# import sys
# import os
# from tqdm import tqdm
# from tensorflow.python import debug as tf_debug
# import json
# import scipy.signal as signal

# from scipy.optimize import linprog
# from scipy.optimize import minimize
# import quadprog
# import scipy
# from scipy.linalg import block_diag

# import pandas as pd
# # Hyper parameter
# from qpsolvers import solve_qp
# from qpsolvers import mosek_solve_qp
# from scipy.sparse import csc_matrix
# import cvxopt
# from cvxopt import matrix
# import sympy
# #Learning_rate = 1e-3
# #initial_stddev = 0.5

# #Training Parameter
# training_batch_size = 128
# training_maximum_iteration = 3001
# TEST_NUM = 2000

# class MRDR_NN(object):
#     def __init__(self, obs_dim, w_hidden, Learning_rate, reg_weight, gamma=1.):

#         self.action_dim = 3
#         self.gamma = gamma
#         self.g = tf.Graph()
#         with self.g.as_default():
#             with tf.variable_scope('mrdr', reuse=False):
#                 self._build_graph(obs_dim, w_hidden, Learning_rate, reg_weight)
#                 self._init_session()
# self.g = tf.Graph()
#     def q_beta(self):
#         return tf.matmul(tf.matrix_diag(self.pi1),  tf.expand_dims(self.out,2)) - tf.expand_dims(self.rew, 2)


#     def _build_graph(self, obs_dim, w_hidden, Learning_rate, reg_weight):
#         # place holder
#         tio2 = tf.placeholder(tf.float32, [None], name='policy_ratio2')


#         self.s = tf.placeholder(tf.float32, [None] + obs_dim, name='state')
#         self.gam = tf.placeholder(tf.float32, [None] , name='gamma_sq')
#         self.omega_cumul = tf.placeholder(tf.float32, [None] , name='cumulative_omega')
#         self.omega = tf.placeholder(tf.float32, [None] , name='current_omega')
#         self.rew = tf.placeholder(tf.float32, [None] + [self.action_dim], name='disc_future_rew')
#         self.pi0 = tf.placeholder(tf.float32, [None] + [self.action_dim], name='pi_b')
#         self.pi1 = tf.placeholder(tf.float32, [None] + [self.action_dim], name='pi_e')


#         # self.factor = self.gam * (self.omega_cumul**2) * self.omega
#         self.Omega = tf.matrix_diag(1/self.pi0)
#         self.Omega = self.Omega - tf.ones_like(self.Omega)

#         self.out = self.Q_val()
#         self.q = self.q_beta()

#         self.loss_pre_reduce = tf.matmul(tf.matmul(tf.transpose(self.q, [0,2,1]) , self.Omega), self.q)
#         self.loss_pre_reduce_w_factor = self.factor * tf.squeeze(self.loss_pre_reduce)
#         self.loss = tf.reduce_sum(self.loss_pre_reduce_w_factor)

#         self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'mrdr'))
#         self.train_op = tf.train.AdamOptimizer(Learning_rate).minimize(self.loss + reg_weight * self.reg_loss)

#         # initialize vars
#         self.init = tf.global_variables_initializer()

#         # Create assign opsfor VAE
#         t_vars = tf.trainable_variables()
#         self.assign_ops = {}
#         for var in t_vars:
#             if var.name.startswith('mrdr'):
#                 pshape = var.get_shape()
#                 pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')
#                 assign_op = var.assign(pl)
#                 self.assign_ops[var] = (assign_op, pl)

#     def _init_session(self):
#         """Launch TensorFlow session and initialize variables"""
#         self.sess = tf.Session(graph=self.g)
#         self.sess.run(self.init)


#     def Q_val(self):
#         with tf.variable_scope('w', reuse = tf.AUTO_REUSE):

#             h = tf.layers.conv2d(self.s, 32, 4, data_format='channels_first', strides=2, activation=tf.nn.relu, name="enc_conv1")
#             h = tf.layers.conv2d(h, 64, 4, data_format='channels_first', strides=2, activation=tf.nn.relu, name="enc_conv2")
#             h = tf.layers.conv2d(h, 128, 4, data_format='channels_first', strides=2, activation=tf.nn.relu, name="enc_conv3")
#             h = tf.layers.conv2d(h, 256, 4, data_format='channels_first', strides=2, activation=tf.nn.relu, name="enc_conv4")
#             h = tf.reshape(h, [-1, 3*5*256])

#             # Some dense layers
#             dense1 = tf.layers.dense(h, 256*3, activation=tf.nn.relu, name="dense1")#, kernel_regularizer = tf.contrib.layers.l2_regularizer(1.), bias_regularizer = regularizer = tf.contrib.layers.l2_regularizer(1.))
#             dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu, name="dense2")#, kernel_regularizer = tf.contrib.layers.l2_regularizer(1.), bias_regularizer = regularizer = tf.contrib.layers.l2_regularizer(1.))
#             out = tf.layers.dense(dense2, self.action_dim, name="Q")

#             return out

#     def get_density_ratio(self, env, states, starts, batch_size=256):
#         bs = batch_size
#         num_batches = int(np.ceil(len(states) / bs))
#         density = []
#         for batch_num in tqdm(range(num_batches)):
#             low_ = batch_num * bs
#             high_ = (batch_num + 1) * bs
#             s = env.pos_to_image(states[low_:high_])

#             out=self.sess.run(self.output, feed_dict = {
#                 self.state : s,
#                 self.isStart: starts
#                 })

#             density.append(out)
#         return np.hstack(density)

#     def _train(self, env, dataset, batch_size = training_batch_size, max_iteration = training_maximum_iteration, test_num = TEST_NUM, fPlot = False, epsilon = 1e-3):




#         # PI0 = np.vstack([x['base_propensity'] for x in dataset])
#         # PI1 = np.vstack([x['target_propensity'] for x in dataset])
#         # REW = np.hstack([x['r'] for x in dataset]).T.reshape(-1)
#         # ACTS = np.hstack([x['a'] for x in dataset]).reshape(-1)
#         # ISSTART = np.hstack([ np.hstack([1] + [0]*(len(x['x'])-1)) for x in dataset])
#         # PI0 = PI0[np.arange(len(ACTS)), ACTS]
#         # PI1 = PI1[np.arange(len(ACTS)), ACTS]



#         PI0 = [x['base_propensity'] for x in dataset]
#         PI1 = [x['target_propensity'] for x in dataset]
#         ACTS = [x['a'] for x in dataset]
#         REW = [x['r'] for x in dataset]
#         omega = [np.array(pi1)[range(len(a)), a] /np.array(pi0)[range(len(a)), a] for pi0,pi1,a in zip(PI0, PI1, ACTS)]
#         gamma = [[self.gamma**(2*t) for t in range(len(a))] for a in ACTS]

#         R = []
#         for traj_num, rew in enumerate(REW):
#             disc_sum = []
#             for i in range(len(rew)):
#                 disc_sum.append(self.discounted_sum(np.array(rew[i:])*np.cumprod(np.hstack([1, omega[traj_num][(i+1):] ])) , self.gamma))
#             R.append(disc_sum)

#         omega_cumul = np.hstack([ np.cumprod( np.hstack([1,om[:-1] ]) ) for om in omega])[np.newaxis,...]

#         factor = np.hstack(gamma)[np.newaxis,...] * omega_cumul**2 * np.hstack(omega)[np.newaxis,...]

#         S = np.vstack([x['x'] for x in dataset])
#         SN = np.vstack([x['x_prime'] for x in dataset])
#         R = np.hstack(R)
#         ACTS = np.hstack(ACTS)
#         PI0 = np.vstack(PI0)
#         PI1 = np.vstack(PI1)
#         # omega_cumul = np.hstack([np.cumprod(om) for om in omega])
#         omega = np.hstack(omega)
#         gamma = np.hstack(gamma)

#         for i in range(max_iteration):
#             # if test_num > 0 and i % 100 == 0:
#             #     subsamples = np.random.choice(test_num, batch_size)
#             #     s_test = env.pos_to_image(S_test[subsamples])
#             #     sn_test = env.pos_to_image(SN_test[subsamples])
#             #     # policy_ratio_test = POLICY_RATIO_test[subsamples]
#             #     policy_ratio_test = (PI1_test[subsamples] + epsilon)/(PI0_test[subsamples] + epsilon)

#             #     subsamples = np.random.choice(test_num, batch_size)
#             #     s_test2 = env.pos_to_image(S_test[subsamples])
#             #     sn_test2 = env.pos_to_image(SN_test[subsamples])
#             #     # policy_ratio_test2 = POLICY_RATIO_test[subsamples]
#             #     policy_ratio_test2 = (PI1_test[subsamples] + epsilon)/(PI0_test[subsamples] + epsilon)
#             #     start = ISSTART_test[subsamples]
#             #     # loss_xx, K_xx, diff_xx, left, x, x2, norm_K = self.sess.run([self.loss_xx, self.K_xx, self.diff_xx, self.left, self.x, self.x2, self.norm_K], feed_dict = {self.med_dist: med_dist,self.state: s_test,self.next_state: sn_test,self.policy_ratio: policy_ratio_test,self.state2: s_test2,self.next_state2: sn_test2,self.policy_ratio2: policy_ratio_test2})
#             #     # import pdb; pdb.set_trace()
#             #     test_loss, reg_loss, norm_w, norm_w_next = self.sess.run([self.loss,
#             #                                                         self.reg_loss,
#             #                                                         self.debug1,
#             #                                                         self.debug2],
#             #                                         feed_dict = {self.med_dist:
#             #                                                             med_dist,
#             #                                                      self.state:
#             #                                                             s_test,
#             #                                                      self.next_state:
#             #                                                             sn_test,
#             #                                                      self.policy_ratio:
#             #                                                             policy_ratio_test,
#             #                                                     self.state2:
#             #                                                             s_test2,
#             #                                                     self.next_state2:
#             #                                                             sn_test2,
#             #                                                     self.policy_ratio2:
#             #                                                             policy_ratio_test2,
#             #                                                     self.isStart:
#             #                                                             start})

#             #     print('----Iteration = {}-----'.format(i))
#             #     print("Testing error = {}".format(test_loss))
#             #     print('Regularization loss = {}'.format(reg_loss))
#             #     print('Norm_w = {}'.format(norm_w))
#             #     print('Norm_w_next = {}'.format(norm_w_next))
#             #     DENR = self.get_density_ratio(env, S_test, ISSTART_test)
#             #     # T = DENR*POLICY_RATIO2
#             #     T = DENR*PI1_test/PI0_test
#             #     # print('DENR = {}'.format(np.sum(T*REW_test)/np.sum(T)))
#             #     num_traj = sum(ISSTART_test)
#             #     print('DENR = {}'.format(np.sum(T*REW_test)/num_traj))
#             #     sys.stdout.flush()
#             #     # epsilon *= 0.9

#             subsamples = np.random.choice(len(S), batch_size)
#             s = env.pos_to_image(S[subsamples])
#             acts = ACTS[subsamples]

#             # _, loss =self.sess.run([self.train_op, self.loss], feed_dict = {
#             #     self.s: s,
#             #     self.gam: gam,
#             #     self.omega_cumul: omega_cumul,
#             #     self.omega: omega,
#             #     self.rew: rew,
#             #     self.pi0: pi0,
#             #     self.pi1: pi1,
#             #     self.acts: acts,
#             #     })
#             _, loss, loss_pre_reduce =self.sess.run([self.train_op, self.loss, self.loss_pre_reduce ], feed_dict = {
#             # out =self.sess.run([self.db1, self.db2], feed_dict = {
#                 self.s: s,
#                 self.gam: gamma[subsamples],
#                 self.omega_cumul: omega_cumul[subsamples],
#                 self.omega: omega[subsamples],
#                 self.rew: np.eye(self.action_dim)[acts] * R[subsamples][...,np.newaxis],
#                 self.pi0: PI0[subsamples],
#                 self.pi1: PI1[subsamples],
#                 self.factor: factor[subsamples]
#                 })
#             print(i, max_iteration, loss)
#         # DENR = self.get_density_ratio(env, S)
#         # # T = DENR*POLICY_RATIO2
#         # T = DENR*PI1/PI0

#         # return np.sum(T*REW)/np.sum(T)

#     def create(self, env, dataset, filename):
#         # S = []
#         # POLICY_RATIO = []
#         # REW = []
#         # for sasr in SASR0:
#         #   for state, action, next_state, reward in sasr:
#         #       POLICY_RATIO.append(policy1.pi(state, action)/policy0.pi(state, action))
#         #       S.append(state)
#         #       REW.append(reward)


#         path = os.path.join(filename, 'mrdr')
#         if os.path.isfile(path):
#             print('Loading MRDR model')
#             self.load_json(path)
#         else:
#             print('Training MRDR model')
#             _ = self._train(env, dataset)
#             self.save_json(path)

#     def evaluate(self, env, dataset):

#         S = np.vstack([x['x'] for x in dataset])
#         SN = np.vstack([x['x_prime'] for x in dataset])
#         PI0 = np.vstack([x['base_propensity'] for x in dataset])
#         PI1 = np.vstack([x['target_propensity'] for x in dataset])
#         REW = np.hstack([x['r'] for x in dataset]).T.reshape(-1)
#         ACTS = np.hstack([x['a'] for x in dataset]).reshape(-1)
#         ISSTART = np.hstack([ np.hstack([1] + [0]*(len(x['x'])-1)) for x in dataset])
#         PI0 = PI0[np.arange(len(ACTS)), ACTS]
#         PI1 = PI1[np.arange(len(ACTS)), ACTS]
#         POLICY_RATIO = PI1/PI0

#         # S = np.array(S)
#         # S_max = np.max(S, axis = 0)
#         # S_min = np.min(S, axis = 0)
#         # S = (S - S_min)/(S_max - S_min)
#         # POLICY_RATIO = np.array(POLICY_RATIO)
#         # REW = np.array(REW)
#         DENR = self.get_density_ratio(env, S, ISSTART)
#         T = DENR*POLICY_RATIO
#         num_traj = sum(ISSTART)

#         return np.sum(T*REW)/num_traj#np.sum(T)

#     def Q(self, pi_e, s, act):
#         Q_val =self.sess.run([self.out], feed_dict = {
#                 self.s: s,
#                 })
#         return np.array(Q_val)

#     def get_model_params(self):
#         # get trainable params.
#         model_names = []
#         model_params = []
#         model_shapes = []
#         with self.g.as_default():
#             t_vars = tf.trainable_variables()
#             for var in t_vars:
#                 if var.name.startswith('mrdr'):
#                     param_name = var.name
#                     p = self.sess.run(var)
#                     model_names.append(param_name)
#                     params = np.round(p*10000).astype(np.int).tolist()
#                     model_params.append(params)
#                     model_shapes.append(p.shape)
#         return model_params, model_shapes, model_names
#     def set_model_params(self, params):
#         with self.g.as_default():
#             t_vars = tf.trainable_variables()
#             idx = 0
#             for var in t_vars:
#                 if var.name.startswith('mrdr'):
#                     pshape = tuple(var.get_shape().as_list())
#                     p = np.array(params[idx])
#                     assert pshape == p.shape, "inconsistent shape"
#                     assign_op, pl = self.assign_ops[var]
#                     self.sess.run(assign_op, feed_dict={pl.name: p/10000.})
#                     idx += 1
#     def load_json(self, jsonfile='mrdr.json'):
#         with open(jsonfile, 'r') as f:
#             params = json.load(f)
#         self.set_model_params(params)
#     def save_json(self, jsonfile='mrdr.json'):
#         model_params, model_shapes, model_names = self.get_model_params()
#         qparams = []
#         for p in model_params:
#             qparams.append(p)
#         with open(jsonfile, 'wt') as outfile:
#             json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

#     def discounted_sum(self, costs, discount):
#         '''
#         Calculate discounted sum of costs
#         '''
#         y = signal.lfilter([1], [1, -discount], x=costs[::-1])
#         return y[::-1][0]


# class MRDR_tabular(object):
#     def __init__(self, gamma=1., action_dim=2):
#         self.gamma = gamma
#         self.action_dim = action_dim

#     def discounted_sum(self, costs, discount):
#         '''
#         Calculate discounted sum of costs
#         '''
#         y = signal.lfilter([1], [1, -discount], x=costs[::-1])
#         return y[::-1][0]

#     def run(self, trajectories):
#         assert self.action_dim == 2, 'This b1, below, is only right for action_dim = 2'

#         PI0 = [x['base_propensity'] for x in trajectories]
#         PI1 = [x['target_propensity'] for x in trajectories]
#         ACTS = [x['a'] for x in trajectories]
#         REW = [x['r'] for x in trajectories]
#         omega = [np.array(pi1)[range(len(a)), a] /np.array(pi0)[range(len(a)), a] for pi0,pi1,a in zip(PI0, PI1, ACTS)]
#         gamma = [[self.gamma**(2*t) for t in range(len(a))] for a in ACTS]


#         R = []
#         for traj_num, rew in enumerate(REW):
#             disc_sum = []
#             for i in range(len(rew)):
#                 disc_sum.append(self.discounted_sum(np.array(rew[i:])*np.cumprod(np.hstack([1, omega[traj_num][(i+1):] ])) , self.gamma))
#             R.append(disc_sum)

#         R = np.hstack(R)[np.newaxis,...]
#         omega_cumul = np.hstack([ np.cumprod( np.hstack([1,om[:-1] ]) ) for om in omega])[np.newaxis,...]

#         factor = np.hstack(gamma)[np.newaxis,...] * omega_cumul**2 * np.hstack(omega)[np.newaxis,...]

#         transitions = np.vstack([ factor,
#                                   np.array([x['x'] for x in trajectories]).reshape(-1,1).T ,
#                                   np.array([x['a'] for x in trajectories]).reshape(-1,1).T ,
#                                   R,
#                                   ]).T

#         FSAR, idxs, counts = np.unique(transitions, return_index=True, return_counts=True, axis=0)

#         propensities = np.hstack([ np.vstack([x['base_propensity'] for x in trajectories]),
#                                    np.vstack([x['target_propensity'] for x in trajectories])
#                                    ])
#         propensities = propensities[idxs]
#         base_propensity, target_propensity = propensities[:,:len(propensities[0])//2], propensities[:,len(propensities[0])//2:]



#         df = pd.DataFrame(np.hstack([FSAR, counts[:,None]]), columns=['factor','s','a','r','counts'])#,'s_','d'])

#         self.Q = np.zeros((len(np.unique(df['s'])), self.action_dim))
#         self.mapping = {}
#         for s, group_df in df.groupby(['s']):

#             s = int(s)
#             self.mapping[s] = s
#             rows = group_df.index.values
#             Omega = np.array([np.diag(1/pi0) - 1 for pi0 in base_propensity[rows]])
#             D = np.array([np.diag(pi1) for pi1 in target_propensity[rows]])
#             pi1 = np.array(target_propensity[rows]).reshape(-1)
#             Omega = Omega * group_df['factor'][:,None,None] * group_df['counts'][:, None, None]

#             eps = 1e-16 # slack
#             block_diag_Omega = block_diag(*Omega)
#             # block_diag_Omega = block_diag(*[block_diag_Omega, np.eye((len(Omega)-1)*(self.action_dim))*eps**2])

#             P  = block_diag_Omega*2 #(block_diag_Omega + block_diag_Omega.T)
#             import pdb; pdb.set_trace()
#             P  = scipy.sparse.lil_matrix(P)
#             P  = cvxopt.spmatrix(P.tocoo().data, P.tocoo().row, P.tocoo().col)

#             actions = np.array(group_df['a']).astype(int)
#             rew = np.array(group_df['r'])

#             if len(actions) > 1:


#                 A = scipy.sparse.lil_matrix(scipy.sparse.eye((len(Omega)-1)*(self.action_dim),len(Omega)*self.action_dim ))#+ (len(Omega)-1)*(self.action_dim)))
#                 for row in np.arange(A.shape[0]):
#                     A[row, row] = 1 / pi1[:-2][row]
#                     A[row, row+self.action_dim] = -1 / pi1[2:][row]
#                     # A[row, len(Omega)*self.action_dim+row] = eps


#                 A = cvxopt.spmatrix(A.tocoo().data, A.tocoo().row, A.tocoo().col)


#                 b1 = np.vstack([ np.array([[ (a==0)*(r)  ], [0]]) for (a,a2,r,r2) in  zip(actions[:-1], actions[1:], rew[:-1], rew[1:]) ]) / pi1[:-2][:, None]
#                 b2 = np.vstack([ np.array([[ (a2 == 0)*r2], [0]]) for (a,a2,r,r2) in  zip(actions[:-1], actions[1:], rew[:-1], rew[1:]) ]) / pi1[2:][:, None]
#                 b3 = np.vstack([ np.array([[0], [(a== 1)*(r)]]) for (a,a2,r,r2) in  zip(actions[:-1], actions[1:], rew[:-1], rew[1:]) ])   / pi1[:-2][:, None]
#                 b4 = np.vstack([ np.array([[0], [(a2 == 1)*r2]]) for (a,a2,r,r2) in  zip(actions[:-1], actions[1:], rew[:-1], rew[1:]) ])  / pi1[2:][:, None]

#                 b = -b1 + b2 - b3 + b4

#                 b = cvxopt.matrix(b)

#                 # Solve min_y y^T P y subject to Ay = b, with some added slack variable inside P to stop ill-conditioning
#                 # y = self.sparse_solve(P, A, b)
#                 y = self.quadratic_solver(np.array(matrix(P)), np.array(matrix(A)), np.array(matrix(b)).T[0])
#             else:
#                 y = self.sparse_solve_no_constraint(P)

#             # import pdb; pdb.set_trace()

#             # np.array(matrix(A)).dot(y) - np.array(matrix(b)).reshape(-1)
#             # np.array(matrix(A)).dot(y) - b

#             resid = [y[(i*self.action_dim):((i+1)*self.action_dim)].reshape(-1) + (np.eye(self.action_dim)[actions[i]] * rew[i]) for i in range(len(rew))]


#             Q = [np.linalg.inv(D[i]).dot(np.array(resid)[i]) for i in range(len(D))]
#             self.Q[s] = Q[0]





#             # self.Q[s] = np.linalg.inv(D[i]).dot((y[(i*self.action_dim):((i+1)*self.action_dim)][...,np.newaxis] + np.vstack([ [(actions[i]==0)*rew[i]], [(actions[i]==1)*rew[i]] ]) ) ).T[0]
#             # for i in range(len(D)): print(np.linalg.inv(D[i]).dot((y[(i*self.action_dim):((i+1)*self.action_dim)][...,np.newaxis] + np.vstack([ [(actions[i]==0)*rew[i]], [(actions[i]==1)*rew[i]] ]) ) ))

#         return self.Q, self.mapping


#     @staticmethod
#     def minitest():

#         d1, d2, r1, r2 = 2,-1,3,1
#         Omega = np.array([[5., 0.],[0., 10.]])
#         A = np.array([[1/d1, -1/d2]])
#         b1 = np.array([-r1/d1 + r2/d2])
#         q1 = np.zeros(Omega.shape[0])
#         G = np.zeros(Omega.shape)
#         h1 = np.zeros(Omega.shape[0])
#         y = solve_qp(Omega, q1, G, h1, A, b1)

#         def get_x(f, d, r):return (f + r) / d


#     @staticmethod
#     def return_Q(y, A, b):
#         # y = output of quad program
#         pass

#     @staticmethod
#     def quadratic_solver(Omega, A, b):
#         q = np.zeros(Omega.shape[0])
#         G = np.zeros(Omega.shape)
#         h = np.zeros(Omega.shape[0])
#         y = solve_qp(Omega, q, G, h, A, b)
#         return y

#     @staticmethod
#     def sparse_solve(Omega, A, b):
#         q = cvxopt.matrix(0., (Omega.size[0],1))
#         # G = cvxopt.spmatrix(0., [0], [0], Omega.size)
#         # h = cvxopt.matrix(0., (1,Omega.size[0]))
#         y = cvxopt.solvers.coneqp(Omega, q, A=A, b=b)
#         print(y)
#         return np.array(y['x'])

#     @staticmethod
#     def sparse_solve_no_constraint(Omega):
#         q = cvxopt.matrix(0., (Omega.size[0],1))
#         # G = cvxopt.spmatrix(0., [0], [0], Omega.size)
#         # h = cvxopt.matrix(0., (1,Omega.size[0]))
#         y = cvxopt.solvers.coneqp(Omega, q)
#         print(y)
#         return np.array(y['x'])


