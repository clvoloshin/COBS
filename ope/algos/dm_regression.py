
import sys
import numpy as np
import pandas as pd
sys.path.append("..")
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
from functools import partial


class DirectMethodRegression(object):
    def __init__(self, data, gamma, frameskip=2, frameheight=2, modeltype = 'conv', processor=None):
        self.data = data
        self.gamma = gamma
        self.frameskip = frameskip
        self.frameheight = frameheight
        self.modeltype = modeltype
        self.processor = processor

        # self.setup(deepcopy(self.trajectories))

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

    def run(self, pi_b, pi_e, epsilon=0.001):


        dataset = self.data.all_transitions()
        frames = self.data.frames()
        omega = self.data.omega()
        rewards = self.data.rewards()

        omega = [np.cumprod(om) for om in omega]
        gamma_vec = self.gamma**np.arange(max([len(x) for x in omega]))

        factors, Rs = [], []
        for data in dataset:
            ts = data[-1]
            traj_num = data[-2]

            i,t = int(traj_num), int(ts)
            Rs.append( np.sum( omega[i][t:]/omega[i][t] * gamma_vec[t:]/gamma_vec[t] *  rewards[i][t:] )  )
            factors.append( gamma_vec[t] * omega[i][t] )

        self.alpha = 1
        self.lamb = 1
        self.cond_number_threshold_A = 1
        block_size = len(dataset)



        phi = self.compute_grid_features()
        self.weight = self.wls_sherman_morrison(phi, Rs, factors, self.lamb, self.alpha, self.cond_number_threshold_A, block_size)

        return DMModel(self.weight,
                        self.data)


    def compute_feature_without_time(self, state, action, step):
        T = max(self.data.lengths())
        n_dim = self.data.n_dim
        n_actions = self.data.n_actions

        # feature_dim = n_dim + n_actions
        # feature_dim =
        phi = np.zeros((n_dim, n_actions))
        # for k in range(step, T):
        #     phi[state * n_actions + action] = env.gamma_vec[k - step]

        # phi = np.hstack([np.eye(n_dim)[int(state)] , np.eye(n_actions)[action] ])
        # phi[action*n_dim: (action+1)*n_dim] = state + 1
        # phi[int(state*n_actions + action)] = 1
        phi[int(state), int(action)] = 1
        phi = phi.reshape(-1)

        return phi


    def compute_feature(self, state, action, step):
        return self.compute_feature_without_time(state, action, step)


    def compute_grid_features(self):

        T = max(self.data.lengths())
        n_dim = self.data.n_dim
        n_actions = self.data.n_actions


        n = len(self.data)


        data_dim = n * T

        phi = data_dim * [None]

        lengths = self.data.lengths()
        for i in range(n):
            states = self.data.states(False, i, i+1)
            actions = self.data.actions()[i]

            for t in range(max(lengths)):
                if t < lengths[i]:
                    s = states[t]
                    action = int(actions[t])
                    phi[i * T + t] = self.compute_feature(s, action, t)
                else:
                    phi[i * T + t] = np.zeros(len(phi[0]))

        return np.array(phi, dtype='float')




    @staticmethod
    def build_model(input_size, scope, action_space_dim=3, modeltype='conv'):

        inp = keras.layers.Input(input_size, name='frames')
        actions = keras.layers.Input((action_space_dim,), name='mask')
        factors = keras.layers.Input((1,), name='weights')

        # conv1 = Conv2D(64, kernel_size=16, strides=2, activation='relu', data_format='channels_first')(inp)
        # #pool1 = MaxPool2D(data_format='channels_first')(conv1)
        # conv2 = Conv2D(64, kernel_size=8, strides=2, activation='relu', data_format='channels_first')(conv1)
        # #pool2 = MaxPool2D(data_format='channels_first')(conv2)
        # conv3 = Conv2D(64, kernel_size=4, strides=2, activation='relu', data_format='channels_first')(conv2)
        # #pool3 = MaxPool2D(data_format='channels_first')(conv3)
        # flat = Flatten()(conv3)
        # dense1 = Dense(10, activation='relu')(flat)
        # dense2 = Dense(30, activation='relu')(dense1)
        # out = Dense(action_space_dim, activation='linear', name=scope+ 'all_Q')(dense2)
        # filtered_output = keras.layers.dot([out, actions], axes=1)

        # model = keras.models.Model(input=[inp, actions], output=[filtered_output])

        # all_Q = keras.models.Model(inputs=[inp],
        #                          outputs=model.get_layer(scope + 'all_Q').output)

        # rmsprop = keras.optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=1e-5, decay=0.0)
        # model.compile(loss='mse', optimizer=rmsprop)

        def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=np.random.randint(2**32))
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
            def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=.1, seed=np.random.randint(2**32))
            # flat = Flatten()(inp)
            # dense1 = Dense(256, activation='relu',kernel_initializer=init(), bias_initializer=init())(flat)
            # dense2 = Dense(256, activation='relu',kernel_initializer=init(), bias_initializer=init())(dense1)
            # dense3 = Dense(128, activation='relu',kernel_initializer=init(), bias_initializer=init())(dense2)
            # out = Dense(32, activation='relu', name='out',kernel_initializer=init(), bias_initializer=init())(dense3)
            flat = Flatten()(inp)
            dense1 = Dense(16, activation='relu',kernel_initializer=init(), bias_initializer=init())(flat)
            # dense2 = Dense(256, activation='relu',kernel_initializer=init(), bias_initializer=init())(dense1)
            dense3 = Dense(8, activation='relu',kernel_initializer=init(), bias_initializer=init())(dense1)
            out = Dense(4, activation='relu', name='out',kernel_initializer=init(), bias_initializer=init())(dense3)



        all_actions = Dense(action_space_dim, name=scope + 'all_Q', activation="linear",kernel_initializer=init(), bias_initializer=init())(out)

        output = keras.layers.dot([all_actions, actions], 1)

        model = keras.models.Model(inputs=[inp, actions], outputs=output)

        model1 = keras.models.Model(inputs=[inp, actions, factors], outputs=output)

        all_Q = keras.models.Model(inputs=[inp],
                                 outputs=model.get_layer(scope + 'all_Q').output)

        # rmsprop = keras.optimizers.RMSprop(lr=0.005, rho=0.95, epsilon=1e-08, decay=1e-3)#, clipnorm=1.)
        adam = keras.optimizers.Adam()

        def DMloss(y_true, y_pred, weights):
            return K.sum(weights * K.square(y_pred - y_true))

        weighted_loss = partial(DMloss, weights=factors)



        model1.compile(loss=weighted_loss, optimizer=adam, metrics=['accuracy'])
        # print(model.summary())
        return model1, model, all_Q

    @staticmethod
    def copy_over_to(source, target):
        target.set_weights(source.get_weights())

    @staticmethod
    def weight_change_norm(model, target_model):
        norm_list = []
        number_of_layers = len(model.layers)
        for i in range(number_of_layers):
            model_matrix = model.layers[i].get_weights()
            target_model_matrix = target_model.layers[i].get_weights()
            if len(model_matrix) >0:
                #print "layer ", i, " has shape ", model_matrix[0].shape
                if model_matrix[0].shape[0] > 0:
                    norm_change = np.linalg.norm(model_matrix[0]-target_model_matrix[0])
                    norm_list.append(norm_change)
        return sum(norm_list)*1.0/len(norm_list)

    def run_linear(self, env, pi_b, pi_e, max_epochs, epsilon=.001):

        self.Q_k = LinearRegression()

        states = self.data.states()
        states = states.reshape(-1,np.prod(states.shape[2:]))
        lengths = self.data.lengths()
        omega = self.data.omega()
        rewards = self.data.rewards()
        actions = self.data.actions().reshape(-1)

        omega = [np.cumprod(om) for om in omega]
        gamma_vec = self.gamma**np.arange(max([len(x) for x in omega]))

        factors, Rs = [], []
        for traj_num, ts in enumerate(self.data.ts()):
            for t in ts:
                i,t = int(traj_num), int(t)
                if omega[i][t]:
                    Rs.append( np.sum( omega[i][t:]/omega[i][t] * gamma_vec[t:]/gamma_vec[t] *  rewards[i][t:] )  )
                else:
                    Rs.append( 0 )
                factors.append( gamma_vec[t] * omega[i][t] )

        Rs = np.array(Rs)
        factors = np.array(factors)

        actions = np.eye(self.data.n_actions)[actions]
        return self.Q_k.fit(np.hstack([states, actions]), Rs, factors)


    def run_NN(self, env, pi_b, pi_e, max_epochs, epsilon=0.001):

        self.dim_of_actions = env.n_actions
        self.Q_k = None
        self.Q_k_minus_1 = None

        earlyStopping = EarlyStopping(monitor='val_loss', min_delta=1e-4,  patience=5, verbose=1, mode='min', restore_best_weights=True)
        mcp_save = ModelCheckpoint('dm_regression.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=1e-4, mode='min')

        self.more_callbacks = [earlyStopping, reduce_lr_loss]

        # if self.modeltype == 'conv':
        #     im = self.trajectories.states()[0,0,...] #env.pos_to_image(np.array(self.trajectories[0]['x'][0])[np.newaxis,...])
        # else:
        #     im = np.array(self.trajectories[0]['frames'])[np.array(self.trajectories[0]['x'][0]).astype(int)][np.newaxis,...]
        im = self.data.states()[0]
        if self.processor: im = self.processor(im)
        self.Q_k, self.Q, self.Q_k_all = self.build_model(im.shape[1:], 'Q_k', modeltype=self.modeltype, action_space_dim=env.n_actions)

        print('Training: Model Free')
        losses = []
        for k in tqdm(range(1)):
            batch_size = 32

            dataset_length = self.data.num_tuples()
            perm = np.random.permutation(range(dataset_length))
            eighty_percent_of_set = int(.8*len(perm))
            training_idxs = perm[:eighty_percent_of_set]
            validation_idxs = perm[eighty_percent_of_set:]
            training_steps_per_epoch = int(1.*np.ceil(len(training_idxs)/float(batch_size)))
            validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batch_size)))
            train_gen = self.generator(env, pi_e, training_idxs, fixed_permutation=True, batch_size=batch_size)
            val_gen = self.generator(env, pi_e, validation_idxs, fixed_permutation=True, batch_size=batch_size, is_train=False)

            hist = self.Q_k.fit_generator(train_gen,
                               steps_per_epoch=training_steps_per_epoch,
                               validation_data=val_gen,
                               validation_steps=validation_steps_per_epoch,
                               epochs=max_epochs,
                               max_queue_size=1,
                               workers=1,
                               use_multiprocessing=False,
                               verbose=1,
                               callbacks = self.more_callbacks)

        return self.Q_k, self.Q_k_all

    @threadsafe_generator
    def generator(self, env, pi_e, all_idxs, fixed_permutation=False,  batch_size = 64, is_train=True):
        data_length = len(all_idxs)
        steps = int(np.ceil(data_length/float(batch_size)))

        states = self.data.states()
        states = states.reshape(tuple([-1]) + states.shape[2:])
        lengths = self.data.lengths()
        omega = self.data.omega()
        rewards = self.data.rewards()
        actions = self.data.actions().reshape(-1)

        omega = [np.cumprod(om) for om in omega]
        gamma_vec = self.gamma**np.arange(max([len(x) for x in omega]))

        factors, Rs = [], []
        for traj_num, ts in enumerate(self.data.ts()):
            for t in ts:
                i,t = int(traj_num), int(t)
                if omega[i][t]:
                    Rs.append( np.sum( omega[i][t:]/omega[i][t] * gamma_vec[t:]/gamma_vec[t] *  rewards[i][t:] )  )
                else:
                    Rs.append( 0 )
                factors.append( gamma_vec[t] * omega[i][t] )

        Rs = np.array(Rs)
        factors = np.array(factors)

        dones = self.data.dones()
        alpha = 1.

        # Rebalance dataset
        probs = np.hstack([np.zeros((dones.shape[0],2)), dones,])[:,:-2]
        if np.sum(probs):
            done_probs = probs / np.sum(probs)
            probs = 1 - probs + done_probs
        else:
            probs = 1 - probs
        probs = probs.reshape(-1)
        probs /= np.sum(probs)
        probs = probs[all_idxs]
        probs /= np.sum(probs)

        dones = dones.reshape(-1)

        # if is_train:
        #     while True:
        #         batch_idxs = np.random.choice(all_idxs, batch_size, p = probs)

        #         x = states[batch_idxs]
        #         weight = factors[batch_idxs]
        #         R = Rs[batch_idxs]
        #         acts = actions[batch_idxs]

        #         yield ([x, np.eye(3)[acts], np.array(weight).reshape(-1,1)], [np.array(R).reshape(-1,1)])
        # else:
        #
        while True:
            perm = np.random.permutation(all_idxs)
            for batch in np.arange(steps):
                batch_idxs = perm[(batch*batch_size):((batch+1)*batch_size)]

                x = states[batch_idxs]
                if self.processor: x = self.processor(x)
                weight = factors[batch_idxs] #* probs[batch_idxs]
                R = Rs[batch_idxs]
                acts = actions[batch_idxs]

                yield ([x, np.eye(env.n_actions)[acts], np.array(weight).reshape(-1,1)], [np.array(R).reshape(-1,1)])




class DMModel(object):
    def __init__(self, weights, data):
        self.weights = weights
        self.data = data

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
        else:
            raise

    def compute_feature_without_time(self, state, action, step):
        T = max(self.data.lengths())
        n_dim = self.data.n_dim
        n_actions = self.data.n_actions

        # feature_dim = n_dim * n_actions
        # phi = np.zeros(feature_dim)
        # # for k in range(step, T):
        # #     phi[state * n_actions + action] = env.gamma_vec[k - step]
        # # phi = np.hstack([np.eye(n_dim)[int(state)] , np.eye(n_actions)[action] ])
        # # phi[action*n_dim: (action+1)*n_dim] = 1 #state + 1

        # phi[int(state*n_actions + action)] = 1
        phi = np.zeros((n_dim, n_actions))
        # for k in range(step, T):
        #     phi[state * n_actions + action] = env.gamma_vec[k - step]

        # phi = np.hstack([np.eye(n_dim)[int(state)] , np.eye(n_actions)[action] ])
        # phi[action*n_dim: (action+1)*n_dim] = state + 1
        # phi[int(state*n_actions + action)] = 1
        phi[int(state), int(action)] = 1
        phi = phi.reshape(-1)

        return phi


    def compute_feature(self, state, action, step):
        return self.compute_feature_without_time(state, action, step)


