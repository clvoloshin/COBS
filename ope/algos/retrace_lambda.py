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


class Retrace(object):
    """Algorithm: Retrace Family (Retrace(lambda), Q(lambda), Tree-Backup(lambda)).
    """
    def __init__(self, data, gamma, frameskip=1, frameheight=1, modeltype='linear', lamb=1., processor=None, max_iters=500):
        """
        Parameters
        ----------
        data : obj
            The logging (historial) dataset.
        gamma : float
            Discount factor.
        frameskip : int, optional, deprecated.
            Deprecated.
        frameheight : int, optional, deprecated.
            Deprecated.
        modeltype: str, optional
            The type of model to represent the Q function.
            Default: 'linear'
        lamb : float, optional
            Float between 0 and 1 representing the coefficient lambda in the algorithm
            Default: 1.0
        processor: function, optional
            Receives state as input and converts it into a different form.
            The new form becomes the input to the direct method.
            Default: None
        max_iters: int, optional
            Maximum number of iterations in the algorithm.
            Default: 500
        """
        self.data = data
        self.gamma = gamma
        self.lamb = lamb
        self.frameskip= frameskip
        self.frameheight = frameheight
        self.modeltype = modeltype
        self.processor = processor
        self.max_iters = max_iters


    def run(self, pi_b, pi_e, method, epsilon=0.001, lamb=None, verbose=True, diverging_epsilon=1000):
        """(Tabular) Get the FQE OPE Q function for pi_e.

        Parameters
        ----------
        pi_b : obj
            A policy object, behavior policy.
        pi_e: obj
            A policy object, evaluation policy.
        method : str
            One of 'retrace', 'tree-backup','Q^pi(lambda)', 'IS'
        epsilon : float, optional
            Convergence criteria.
            Default: 0.001
        lamb : float, optional
            Float between 0 and 1 representing the coefficient lambda in the algorithm
            Default: None
        verbose: bool
            Print diagnostics
            Default: True
        diverging_epsilon : int
            Threshold above which algorithm terminates due to divergence
            Default: 1000
        
        Returns
        -------
        float, obj2, obj3
            float: OPE Estimate
            obj2: 2D ndarray Q table. Q[map(s),a]
            obj3: dic, maps state to row in the Q table
        """
        lamb = lamb if lamb is not None else self.lamb
        assert method in ['retrace','tree-backup','Q^pi(lambda)','IS']

        S = np.squeeze(self.data.states())
        SN = np.squeeze(self.data.next_states())
        ACTS = self.data.actions()
        REW = self.data.rewards()
        PIE = self.data.target_propensity()
        PIB = self.data.base_propensity()
        DONES = self.data.dones()

        unique_states = np.unique(np.vstack([S, SN]))
        state_space_dim = len(unique_states)
        action_space_dim = pi_e.action_space_dim

        U1 = np.zeros(shape=(state_space_dim,action_space_dim))

        mapping = {state:idx for idx,state in enumerate(unique_states)}

        state_action_to_idx = {}
        for row,SA in enumerate(zip(S,ACTS)):
            for col, (state,action) in enumerate(zip(*SA)):
                if tuple([state,action]) not in state_action_to_idx: state_action_to_idx[tuple([state,action])] = []
                state_action_to_idx[ tuple([state,action]) ].append([row, col])

        count = 0
        eps = 1e-8
        while True:
            U = U1.copy()
            update = np.zeros(shape=(state_space_dim,action_space_dim))
            delta = 0

            out = []
            for s,a,r,sn,pie,pib,done in zip(S,ACTS,REW,SN,PIE,PIB,DONES):
                t = len(s)
                s = np.array([mapping[s_] for s_ in s])
                sn = np.array([mapping[s_] for s_ in sn])

                if method == 'retrace':
                    c = pie[range(len(a)), a]/(pib[range(len(a)), a] + eps)
                    c[0] = 1.
                    c = lamb * np.minimum(1., c) # c_s = lambda * min(1, pie/pib)
                elif method == 'tree-backup':
                    c = lamb * pie[range(len(a)), a] # c_s = lambda * pi(a|x)
                    c[0] = 1.
                elif method == 'Q^pi(lambda)':
                    c = np.ones_like(a)*lamb # c_s = lambda
                    c[0] = 1.
                elif method == 'IS':
                    c = pie[range(len(a)), a]/(pib[range(len(a)), a] + eps) # c_s = pie/pib
                    c[0] = 1.
                    c = c # c_s = pie/pib
                else:
                    raise

                c = np.cumprod(c)
                gam = self.gamma ** np.arange(t)

                expected_U = np.sum(pi_e.predict(sn)*U[sn, :], axis=1)*(1-done)
                # expected_U = np.sum([], axu

                diff = r + self.gamma * expected_U - U[s, a]

                # import pdb; pdb.set_trace()
                val = gam * c * diff

                out.append(np.cumsum(val[::-1])[::-1])

            out = np.array(out)

            for key, val in state_action_to_idx.items():
                rows, cols = np.array(val)[:,0], np.array(val)[:,1]
                state, action = key[0], key[1]
                state = mapping[state]
                update[state, action] = np.mean(out[rows,cols])

            U1 = U1 + update
            delta = np.linalg.norm(U-U1)
            count += 1
            if verbose: print(count, delta)
            if delta < epsilon or count > self.max_iters or delta > diverging_epsilon:# * (1 - self.gamma) / self.gamma:
                return np.sum([prob*U1[0, new_a] for new_a,prob in enumerate(pi_e.predict([0])[0])]), U1, mapping #U[0,pi_e([0])][0]




    @staticmethod
    def build_model(input_size, scope, action_space_dim=3, modeltype='conv'):
        """Build NN Q function.

        Parameters
        ----------
        input_size : ndarray
            (# Channels, # Height, # Width)
        scope: str
            Name for the NN
        action_space_dim : int, optional
            Action space cardinality. 
            Default: 3
        modeltype : str, optional
            The model type to be built.
            Default: 'conv'
        
        Returns
        -------
        obj1, obj2
            obj1: Compiled model
            obj2: Forward model Q(s) -> R^|A| 
        """

        inp = keras.layers.Input(input_size, name='frames')
        actions = keras.layers.Input((action_space_dim,), name='mask')

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
        elif modeltype == 'mlp':
            def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=.1, seed=np.random.randint(2**32))
            flat = Flatten()(inp)
            dense1 = Dense(16, activation='relu',kernel_initializer=init(), bias_initializer=init())(flat)
            # dense2 = Dense(256, activation='relu',kernel_initializer=init(), bias_initializer=init())(dense1)
            dense3 = Dense(8, activation='relu',kernel_initializer=init(), bias_initializer=init())(dense1)
            out = Dense(4, activation='relu', name='out',kernel_initializer=init(), bias_initializer=init())(dense3)
        elif modeltype == 'linear':
            def init(): return keras.initializers.TruncatedNormal(mean=0.0, stddev=.001, seed=np.random.randint(2**32))
            out = Flatten()(inp)
        else:
            raise NotImplemented


        all_actions = Dense(action_space_dim, name=scope + 'all_Q', activation="linear",kernel_initializer=init(), bias_initializer=init())(out)

        output = keras.layers.dot([all_actions, actions], 1)

        model = keras.models.Model(inputs=[inp, actions], outputs=output)

        all_Q = keras.models.Model(inputs=[inp],
                                 outputs=model.get_layer(scope + 'all_Q').output)

        rmsprop = keras.optimizers.RMSprop(lr=0.05, rho=0.95, epsilon=1e-08, decay=1e-3)#, clipnorm=1.)
        adam = keras.optimizers.Adam(clipnorm=1.)
        model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
        return model, all_Q

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

    # def run_linear(self, env, method, pi_b, pi_e, max_epochs, epsilon=.001, lamb=.5):

    #     lamb = lamb if lamb is not None else self.lamb
    #     assert method in ['retrace','tree-backup','Q^pi(lambda)','IS']

    #     S = np.squeeze(self.data.states())
    #     SN = np.squeeze(self.data.next_states())
    #     ACTS = self.data.actions()
    #     REW = self.data.rewards()
    #     PIE = self.data.target_propensity()
    #     PIB = self.data.base_propensity()
    #     DONES = self.data.dones()

    #     action_space_dim = env.n_actions
    #     ACTS_reshaped = np.eye(action_space_dim)[ACTS.reshape(-1)]

    #     self.Q_k = LinearRegression()
    #     for epoch in tqdm(range(max_epochs)):
    #         X = np.hstack([S.reshape(np.hstack([-1, np.prod(S.shape[2:])])) , ACTS_reshaped])
    #         if epoch > 0:
    #             Q_ = self.Q_k.predict(X)
    #             delta = np.linalg.norm(Q_ - Q)
    #             print(Q_-Q)
    #             print(delta)
    #             Q = Q_
    #         else:
    #             Q = 0
    #         out=[]
    #         for traj_num, (s,a,r,sn,pie,pib,done) in enumerate(zip(S,ACTS,REW,SN,PIE,PIB,DONES)):
    #             t = len(s)
    #             prev_shape = s.shape
    #             s = s.reshape(np.hstack([-1, np.prod(s.shape[1:])]))
    #             sn = sn.reshape(np.hstack([-1, np.prod(sn.shape[1:])]))

    #             if method == 'retrace':
    #                 c = pie[range(len(a)), a]/pib[range(len(a)), a]
    #                 c[0] = 1.
    #                 c = lamb * np.minimum(1., c) # c_s = lambda * min(1, pie/pib)
    #             elif method == 'tree-backup':
    #                 c = lamb * pie[range(len(a)), a] # c_s = lambda * pi(a|x)
    #                 c[0] = 1.
    #             elif method == 'Q^pi(lambda)':
    #                 c = np.ones_like(a)*lamb # c_s = lambda
    #                 c[0] = 1.
    #             elif method == 'IS':
    #                 c = pie[range(len(a)), a]/pib[range(len(a)), a] # c_s = pie/pib
    #                 c[0] = 1.
    #                 c = c # c_s = pie/pib
    #             else:
    #                 raise

    #             c = np.cumprod(c)
    #             gam = self.gamma ** np.arange(t)

    #             if epoch == 0:
    #                 diff = r
    #             else:
    #                 # E_{\pi_e}[Q(x_{t+1}, .)]
    #                 expected_U = np.sum([pi_e.predict(s.reshape(prev_shape))[0][act]*self.Q_k.predict(np.hstack([sn, np.tile(np.eye(action_space_dim)[act], len(sn)).reshape(len(sn),action_space_dim) ]))*(1-done) for act in np.arange(action_space_dim)], axis=0)

    #                 # r + gamma * E_{\pi_e}[Q(x_{t+1}, .)]  - Q(x_t, a_t)
    #                 diff = r + self.gamma * expected_U - self.Q_k.predict(np.hstack([s, np.eye(action_space_dim)[a]]))  # Q.reshape(REW.shape)[traj_num].reshape(-1)#

    #             # import pdb; pdb.set_trace()
    #             val = gam * c * diff

    #             out.append(np.cumsum(val[::-1])[::-1])

    #         out = np.array(out)

    #         self.Q_k.fit(X, Q + out.reshape(-1))

    #     return self.Q_k


    def run_NN(self, env, pi_b, pi_e, max_epochs, method, epsilon=0.001):
        """(Neural) Get the Retrace/Tree/Q^pi OPE Q model for pi_e.

        Parameters
        ----------
        env : obj
            The environment object.
        pi_b : obj
            A policy object, behavior policy.
        pi_e: obj
            A policy object, evaluation policy.
        max_epochs : int
            Maximum number of NN epochs to run
        method : str
            One of 'retrace', 'tree-backup','Q^pi(lambda)', 'IS'
        epsilon : float, optional
            Default: 0.001
        
        Returns
        -------
        float, obj1, obj2
            float: represents the average value of the final 10 iterations
            obj1: Fitted Forward model Q(s,a) -> R
            obj2: Fitted Forward model Q(s) -> R^|A| 
        """
        initial_states = self.data.initial_states()
        if self.processor: initial_states = self.processor(initial_states)
        self.dim_of_actions = env.n_actions
        self.Q_k = None
        self.Q_k_all = None
        self.Q_k_minus_1 = None
        self.Q_k_minus_1_all = None


        # earlyStopping = EarlyStopping(monitor='val_loss', min_delta=1e-4,  patience=10, verbose=1, mode='min', restore_best_weights=True)
        # mcp_save = ModelCheckpoint('fqe.hdf5', save_best_only=True, monitor='val_loss', mode='min')
        # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')

        self.more_callbacks = [] #[earlyStopping, mcp_save, reduce_lr_loss]

        # if self.modeltype == 'conv':
        #     im = env.pos_to_image(np.array(self.trajectories[0]['x'][0])[np.newaxis,...])
        # else:
        #     im = np.array(self.trajectories[0]['frames'])[np.array(self.trajectories[0]['x'][0]).astype(int)][np.newaxis,...]

        im = self.data.states()[0]
        if self.processor: im = self.processor(im)
        self.Q_k, self.Q_k_all = self.build_model(im.shape[1:], 'Q_k', modeltype=self.modeltype, action_space_dim=env.n_actions)
        self.Q_k_minus_1, self.Q_k_minus_1_all = self.build_model(im.shape[1:], 'Q_k_minus_1', modeltype=self.modeltype, action_space_dim=env.n_actions)

        # print('testing Q_k:', )
        tmp_act = np.eye(env.n_actions)[[0]]
        self.Q_k.predict([[im[0]], tmp_act])
        # print('testing Q_k all:', )
        self.Q_k_all.predict([[im[0]]])
        # print('testing Q_k_minus_1:', )
        self.Q_k_minus_1.predict([[im[0]], tmp_act])
        # print('testing Q_k_minus_1 all:', )
        self.Q_k_minus_1_all.predict([[im[0]]])


        self.copy_over_to(self.Q_k, self.Q_k_minus_1)
        values = []

        print('Training: %s' % method)
        losses = []
        for k in tqdm(range(max_epochs)):
            batch_size = 4

            dataset_length = self.data.num_tuples()
            perm = np.random.permutation(range(dataset_length))
            eighty_percent_of_set = int(1.*len(perm))
            training_idxs = perm[:eighty_percent_of_set]
            validation_idxs = perm[eighty_percent_of_set:]
            training_steps_per_epoch = max(500, int(.03 * np.ceil(len(training_idxs)/float(batch_size))))
            validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batch_size)))
            # steps_per_epoch = 1 #int(np.ceil(len(dataset)/float(batch_size)))
            train_gen = self.generator(env, pi_e, training_idxs, method, fixed_permutation=True, batch_size=batch_size)
            # val_gen = self.generator(policy, dataset, validation_idxs, method, fixed_permutation=True, batch_size=batch_size)

            # import pdb; pdb.set_trace()
            # train_gen = self.generator(env, pi_e, (transitions,frames), training_idxs, fixed_permutation=True, batch_size=batch_size)
            # inp, out = next(train_gen)
            M = 5
            hist = self.Q_k.fit_generator(train_gen,
                               steps_per_epoch=training_steps_per_epoch,
                               #validation_data=val_gen,
                               #validation_steps=validation_steps_per_epoch,
                               epochs=1,
                               max_queue_size=50,
                               workers=2,
                               use_multiprocessing=False,
                               verbose=1,
                               callbacks = self.more_callbacks)

            norm_change = self.weight_change_norm(self.Q_k, self.Q_k_minus_1)
            self.copy_over_to(self.Q_k, self.Q_k_minus_1)

            losses.append(hist.history['loss'])
            actions = pi_e.sample(initial_states)
            assert len(actions) == initial_states.shape[0]
            Q_val = self.Q_k_all.predict(initial_states)[np.arange(len(actions)), actions]
            values.append(np.mean(Q_val))
            print(values[-1], norm_change, np.mean(values[-M:]), np.abs(np.mean(values[-M:])- np.mean(values[-(M+1):-1])), 1e-4*np.abs(np.mean(values[-(M+1):-1])))
            if k>M and np.abs(np.mean(values[-M:]) - np.mean(values[-(M+1):-1])) < 1e-4*np.abs(np.mean(values[-(M+1):-1])):
                break

        return np.mean(values[-10:]), self.Q_k, self.Q_k_all
        # actions = policy(initial_states[:,np.newaxis,...], x_preprocessed=True)
        # Q_val = self.Q_k.all_actions([initial_states], x_preprocessed=True)[np.arange(len(actions)), actions]
        # return np.mean(Q_val)*dataset.scale, values

    @threadsafe_generator
    def generator(self, env, pi_e, all_idxs, method, fixed_permutation=False,  batch_size = 64):
        """Data Generator for fitting FQE model

        Parameters
        ----------
        env : obj
            The environment object.
        pi_e: obj
            A policy object, evaluation policy.
        all_idxs : ndarray
            1D array of ints representing valid datapoints from which we generate examples
        method : str
            One of 'retrace', 'tree-backup','Q^pi(lambda)', 'IS'
        fixed_permutation : bool, optional
            Run through the data the same way every time?
            Default: False
        batch_size : int, optional
            Minibatch size to during training
            Default: 64

        
        Yield
        -------
        obj1, obj2
            obj1: [state, action]
            obj2: [Q]
        """

        # dataset, frames = dataset
        data_length = len(all_idxs)
        steps = int(np.ceil(data_length/float(batch_size)))

        states = self.data.states()
        # states = states.reshape(-1,np.prod(states.shape[2:]))
        actions = self.data.actions()
        # actions = np.eye(env.n_actions)[actions]

        next_states = self.data.next_states()
        original_shape = next_states.shape
        next_states = next_states.reshape(-1,np.prod(next_states.shape[2:]))

        pi1_ = self.data.next_target_propensity()
        pi1 = self.data.target_propensity()
        pi0 = self.data.base_propensity()
        rewards = self.data.rewards()

        dones = self.data.dones()
        alpha = 1.

        # balance dataset since majority of dataset is absorbing state
        probs = np.hstack([np.zeros((dones.shape[0],2)), dones,])[:,:-2]
        if np.sum(probs):
            done_probs = probs / np.sum(probs)
            probs = 1 - probs + done_probs
        else:
            probs = 1 - probs
        probs = probs.reshape(-1)
        probs /= np.sum(probs)
        # probs = probs[all_idxs]

        while True:
            batch_idxs = np.random.choice(all_idxs, batch_size, p = probs)
            Ss = []
            As = []
            Ys = []
            for idx in batch_idxs:

                traj_num = int(idx/ self.data.lengths()[0]) # Assume fixed length, horizon is fixed
                i = idx - traj_num * self.data.lengths()[0]
                s = self.data.states(low_=traj_num, high_=traj_num+1)[0,i:]
                sn = self.data.next_states(low_=traj_num, high_=traj_num+1)[0,i:]
                a = actions[traj_num][i:]
                r = rewards[traj_num][i:]
                pie = pi1[traj_num][i:]
                pie_ = pi1_[traj_num][i:]
                pib = pi0[traj_num][i:]

                if method == 'retrace':
                    c = pie[range(len(a)), a]/pib[range(len(a)), a]
                    c[0] = 1.
                    c = self.lamb * np.minimum(1., c) # c_s = lambda * min(1, pie/pib)
                elif method == 'tree-backup':
                    c = self.lamb * pie[range(len(a)), a] # c_s = lambda * pi(a|x)
                    c[0] = 1.
                elif method == 'Q^pi(lambda)':
                    c = np.ones_like(a)*self.lamb # c_s = lambda
                    c[0] = 1.
                elif method == 'IS':
                    c = pie[range(len(a)), a]/pib[range(len(a)), a] # c_s = pie/pib
                    c[0] = 1.
                    c = c # c_s = pie/pib
                else:
                    raise

                c = np.cumprod(c)
                gam = self.gamma ** np.arange(len(s))

                if self.processor:
                    s = self.processor(s)
                    sn = self.processor(sn)

                Q_x = self.Q_k_minus_1_all.predict(s)
                Q_x_ = self.Q_k_minus_1_all.predict(sn)

                Q_xt_at = Q_x[range(len(a)), a]
                E_Q_x_ = np.sum(Q_x_*pie_, axis=1)

                Ss.append(s[0])
                As.append(a[0])
                Ys.append(Q_xt_at[0] + np.sum(gam * c * (r + self.gamma * E_Q_x_ - Q_xt_at)))

            yield [np.array(Ss), np.eye(env.n_actions)[np.array(As)]], [np.array(Ys)]

