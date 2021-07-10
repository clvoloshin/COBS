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
from collections import Counter

class DataHolder(object):
    """Data Structure to hold a transition of data.
    """
    def __init__(self, s, a, r, s_, d, policy_action, original_shape):
        self.states = s
        self.next_states = s_
        self.actions = a
        self.rewards = r
        self.dones = d
        self.policy_action = policy_action
        self.original_shape = original_shape


class FittedQEvaluation(object):
    """Algorithm: Fitted Q Evaluation (FQE).
    """
    def __init__(self, data, gamma, frameskip=2, frameheight=2, modeltype = 'conv', processor=None):
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
            Default: 'conv'
        processor: function, optional
            Receives state as input and converts it into a different form.
            The new form becomes the input to the direct method.
            Default: None
        """
        self.data = data
        self.gamma = gamma
        self.frameskip = frameskip
        self.frameheight = frameheight
        self.modeltype = modeltype
        self.processor = processor

        # self.setup(deepcopy(self.trajectories))

    def setup(self, dataset):
        '''
        '''
        transitions = np.vstack([ np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                  np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
                                  np.array([x['x_prime'] for x in dataset]).reshape(-1,1).T
                                 ]).T



        unique, idx, count = np.unique(transitions, return_index=True, return_counts=True, axis=0)

        partial_transitions = np.vstack([ np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                  np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
                                 ]).T
        unique_a_given_x, idx_a_given_x, count_a_given_x = np.unique(partial_transitions, return_index=True, return_counts=True, axis=0)

        # key=(state, action). value= number of times a was taking in state
        all_counts_a_given_x = {tuple(key):value for key,value in zip(unique_a_given_x,count_a_given_x)}

        prob = {}
        for idx,row in enumerate(unique):
            if tuple(row[:-1]) in prob:
                prob[tuple(row[:-1])][row[-1]] = count[idx] / all_counts_a_given_x[(row[0],row[1])]
            else:
                prob[tuple(row[:-1])] = {}
                prob[tuple(row[:-1])][row[-1]] = count[idx] / all_counts_a_given_x[(row[0],row[1])]

        all_transitions = np.vstack([ np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                      np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
                                      np.array([x['x_prime'] for x in dataset]).reshape(-1,1).T ,
                                      np.array([x['done'] for x in dataset]).reshape(-1,1).T ,
                                 ]).T
        self.terminal_transitions = {tuple([x,a,x_prime]):1 for x,a,x_prime in all_transitions[all_transitions[:,-1] == True][:,:-1]}


        self.P = prob

        transitions = np.vstack([ np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                  np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
                                  # np.array([x['x_prime'] for x in dataset]).reshape(-1,1).T,
                                  np.array([x['r'] for x in dataset]).reshape(-1,1).T ,
                                 ]).T
        unique, idxs, counts = np.unique(transitions, return_index=True, return_counts=True, axis=0)

        partial_transitions = np.vstack([ np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                  np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
                                  # np.array([x['x_prime'] for x in dataset]).reshape(-1,1).T,
                                 ]).T
        unique_a_given_x, idx_a_given_x, count_a_given_x = np.unique(partial_transitions, return_index=True, return_counts=True, axis=0)

        # key=(state, action). value= number of times a was taking in state
        all_counts_a_given_x = {tuple(key):value for key,value in zip(unique_a_given_x,count_a_given_x)}

        rew = {}
        for idx,row in enumerate(unique):
            if tuple(row[:-1]) in rew:
                rew[tuple(row[:-1])][row[-1]] = counts[idx] / all_counts_a_given_x[tuple(row[:-1])]
            else:
                rew[tuple(row[:-1])] = {}
                rew[tuple(row[:-1])][row[-1]] = counts[idx] / all_counts_a_given_x[tuple(row[:-1])]

        self.R = rew

        transitions = np.vstack([ np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                  np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
                                  np.array([x['x_prime'] for x in dataset]).reshape(-1,1).T,
                                  np.array([range(len(x['x'])) for x in dataset]).reshape(-1,1).T,
                                  np.array([x['r'] for x in dataset]).reshape(-1,1).T ,
                                 ]).T
        unique, idxs, counts = np.unique(transitions, return_index=True, return_counts=True, axis=0)

        partial_transitions = np.vstack([ np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                  np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
                                  np.array([x['x_prime'] for x in dataset]).reshape(-1,1).T,
                                  np.array([range(len(x['x'])) for x in dataset]).reshape(-1,1).T,
                                 ]).T
        unique_a_given_x, idx_a_given_x, count_a_given_x = np.unique(partial_transitions, return_index=True, return_counts=True, axis=0)

        # key=(state, action). value= number of times a was taking in state
        all_counts_a_given_x = {tuple(key):value for key,value in zip(unique_a_given_x,count_a_given_x)}


        rew = {}
        for idx,row in enumerate(unique):
            if tuple(row[:-2]) in rew:
                if row[-2] in rew[tuple(row[:-2])]:
                    rew[tuple(row[:-2])][row[-2]][row[-1]] = counts[idx] / all_counts_a_given_x[tuple(row[:-1])]
                else:
                    rew[tuple(row[:-2])][row[-2]] = {}
                    rew[tuple(row[:-2])][row[-2]][row[-1]] = counts[idx] / all_counts_a_given_x[tuple(row[:-1])]
            else:
                rew[tuple(row[:-2])] = {}
                rew[tuple(row[:-2])][row[-2]] = {}
                rew[tuple(row[:-2])][row[-2]][row[-1]] = counts[idx] / all_counts_a_given_x[tuple(row[:-1])]

        self.R1 = rew

    def run(self, pi_b, pi_e, epsilon=0.001, max_epochs=10000, verbose = True):
        """(Tabular) Get the FQE OPE Q function for pi_e.

        Parameters
        ----------
        pi_b : obj
            A policy object, behavior policy.
        pi_e: obj
            A policy object, evaluation policy.
        epsilon : float, optional
            Convergence criteria.
            Default: 0.001
        max_epochs : int, optional
            Max number of iterations
            Default: 10000
        verbose: bool, optional
            Print diagnostics
            Default: True
        
        Returns
        -------
        obj1, obj2, obj3
            obj1: None
            obj2: 2D ndarray Q table. Q[map(s),a]
            obj3: dic, maps state to row in the Q table
        """

        data = self.data.basic_transitions()

        action_space_dim = pi_b.action_space_dim
        state_space_dim = len(np.unique(data[:,[0,3]].reshape(-1)))
        # L = max(data[:,-1]) + 1

        mapping = {state:idx for idx,state in enumerate(np.unique(data[:,[0,3]].reshape(-1)))}

        U1 = np.zeros(shape=(state_space_dim,action_space_dim))
        # print('Num unique in FQE: ', data.shape[0])

        df = pd.DataFrame(data, columns=['x','a','t','x_prime','r','done'])
        initial_states = Counter(df[df['t']==0]['x'])
        total = sum(initial_states.values())
        initial_states = {key:val/total for key,val in initial_states.items()}

        count = -1
        while True:
            U = U1.copy()
            delta = 0
            count += 1

            for (x,a), group in df.groupby(['x','a']):
                x,a = int(x), int(a)
                x = mapping[x]

                # expected_reward = np.mean(group['r'])
                # expected_Q = np.mean([[pi_e.predict([x_prime])[act]*U[x_prime,act] for x_prime in group['x_prime']] for act in range(action_space_dim)])

                vals = np.zeros(group['x_prime'].shape)

                x_primes = np.array([mapping[key] for key in group['x_prime']])
                vals = np.array(group['r']) + self.gamma * np.sum(pi_e.predict(x_primes)*U[x_primes, :], axis=1)*(1-np.array(group['done']))
                # for act in range(action_space_dim):
                #     try:

                #         vals += self.gamma*pi_e.predict(np.array(group['x_prime']))[range(len(x_primes)), act ]*U[x_primes,act]*(1-group['done'])
                #     except:
                #         import pdb; pdb.set_trace()

                # vals += group['r']

                U1[x, a] = np.mean(vals)#expected_reward + self.gamma*expected_Q

                delta = max(delta, abs(U1[x,a] - U[x,a]))
            if verbose: print(count, delta)

            if self.gamma == 1:
                # TODO: include initial state distribution
                if delta < epsilon:
                    out = np.sum([prob*U1[0, new_a] for new_a,prob in enumerate(pi_e.predict([0])[0])]) #U[0,pi_e([0])][0]
                    return None, U1, mapping
                    # return out, U1, mapping
            else:
                if delta < epsilon * (1 - self.gamma) / self.gamma or count>max_epochs:
                    return None, U1, mapping #U[0,pi_e([0])][0]
                    # return np.sum([prob*U1[mapping[0], new_a] for new_a,prob in enumerate(pi_e.predict([0])[0])]), U1, mapping #U[0,pi_e([0])][0]

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
            flat = Flatten()(inp)
            dense1 = Dense(64, activation='elu',kernel_initializer=init(), bias_initializer=init())(flat)
            # dense2 = Dense(256, activation='relu',kernel_initializer=init(), bias_initializer=init())(dense1)
            dense3 = Dense(32, activation='elu',kernel_initializer=init(), bias_initializer=init())(dense1)
            out = Dense(8, activation='elu', name='out',kernel_initializer=init(), bias_initializer=init())(dense3)


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

    def run_linear(self, env, pi_b, pi_e, max_epochs, epsilon=.001, fit_intercept=True):
        """(Linear) Get the FQE OPE estimate.

        Parameters
        ----------
        env : obj
            The environment object.
        pi_b : obj
            A policy object, behavior policy.
        pi_e: obj
            A policy object, evaluation policy.
        max_epochs : int
            Max number of iterations
        epsilon : float
            Convergence criteria.
            Default: 0.001
        fit_intercept : bool
            Fit the y-intercept
            Default: True
        
        Returns
        -------
        obj
            sklearn LinearReg object representing the Q function
        """
        initial_states = self.data.initial_states()
        self.Q_k = LinearRegression(fit_intercept=fit_intercept)
        values = []

        states = self.data.states()
        states = states.reshape(-1,np.prod(states.shape[2:]))
        actions = self.data.actions().reshape(-1)
        actions = np.eye(env.n_actions)[actions]
        X = np.hstack([states, actions])

        next_states = self.data.next_states()
        next_states = next_states.reshape(-1,np.prod(next_states.shape[2:]))

        policy_action = self.data.target_propensity()
        lengths = self.data.lengths()
        omega = self.data.omega()
        rewards = self.data.rewards()

        not_dones = 1-self.data.dones()

        for epoch in tqdm(range(max_epochs)):

            if epoch:
                inp = np.repeat(next_states, env.n_actions, axis=0)
                act = np.tile(np.arange(env.n_actions), len(next_states))
                inp = np.hstack([inp.reshape(inp.shape[0],-1), np.eye(env.n_actions)[act]])
                Q_val = self.Q_k.predict(inp).reshape(policy_action.shape)
            else:
                Q_val = np.zeros_like(policy_action)
            Q = rewards + self.gamma * (Q_val * policy_action).sum(axis=-1) * not_dones
            Q = Q.reshape(-1)

            self.Q_k.fit(X, Q)

            # Check if converged
            actions = pi_e.sample(initial_states)
            Q_val = self.Q_k.predict(np.hstack([initial_states.reshape(initial_states.shape[0],-1), np.eye(env.n_actions)[actions]]))
            values.append(np.mean(Q_val))
            M = 20
            # print(values[-1], np.mean(values[-M:]), np.abs(np.mean(values[-M:])- np.mean(values[-(M+1):-1])), 1e-4*np.abs(np.mean(values[-(M+1):-1])))
            if epoch>M and np.abs(np.mean(values[-M:]) - np.mean(values[-(M+1):-1])) < 1e-4*np.abs(np.mean(values[-(M+1):-1])):
                break
        #np.mean(values[-10:]), self.Q_k,
        return self.Q_k

    def run_linear_value_iter(self, env, pi_b, pi_e, max_epochs, epsilon=.001):
        initial_states = self.data.initial_states()
        self.Q_k = LinearRegression()
        values = []

        states = self.data.states()
        states = states.reshape(-1,np.prod(states.shape[2:]))
        actions = self.data.actions().reshape(-1)
        actions = np.eye(env.n_actions)[actions]
        X = states #np.hstack([states, actions])

        next_states = self.data.next_states()
        next_states = next_states.reshape(-1,np.prod(next_states.shape[2:]))

        policy_action = self.data.target_propensity()
        lengths = self.data.lengths()
        omega = self.data.omega()
        rewards = self.data.rewards()

        not_dones = 1-self.data.dones()

        for epoch in tqdm(range(max_epochs)):


            if epoch:
                # inp = np.repeat(next_states, env.n_actions, axis=0)
                inp = next_states
                # act = np.tile(np.arange(env.n_actions), len(next_states))
                inp = inp.reshape(inp.shape[0],-1) #np.hstack([inp.reshape(inp.shape[0],-1), np.eye(env.n_actions)[act]])
                Q_val = self.Q_k.predict(inp).reshape(policy_action[...,0].shape)
            else:
                Q_val = np.zeros_like(policy_action[...,0]) + 1
            Q = rewards + self.gamma * Q_val * not_dones
            Q = Q.reshape(-1)

            self.Q_k.fit(X, Q)

            # Check if converged
            actions = pi_e.sample(initial_states)
            # Q_val = self.Q_k.predict(np.hstack([initial_states.reshape(initial_states.shape[0],-1), np.eye(env.n_actions)[actions]]))
            Q_val = self.Q_k.predict(initial_states.reshape(initial_states.shape[0],-1)) #self.Q_k.predict(np.hstack([, np.eye(env.n_actions)[actions]]))
            values.append(np.mean(Q_val))
            M = 20
            # print(values[-1], np.mean(values[-M:]), np.abs(np.mean(values[-M:])- np.mean(values[-(M+1):-1])), 1e-4*np.abs(np.mean(values[-(M+1):-1])))
            print(self.Q_k.coef_)
            if epoch>M and np.abs(np.mean(values[-M:]) - np.mean(values[-(M+1):-1])) < 1e-4*np.abs(np.mean(values[-(M+1):-1])):
                break

        #np.mean(values[-10:]), self.Q_k,
        return self.Q_k


    def run_NN(self, env, pi_b, pi_e, max_epochs, epsilon=0.001, perc_of_dataset = 1.):
        """(Neural) Get the FQE OPE Q model for pi_e.

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
        epsilon : float, optional
            Default: 0.001
        perc_of_dataset : float, optional
            How much of the dataset to use for training
            Default: 1.

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
        self.Q_k_minus_1 = None

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

        # policy_action = np.vstack([episode['target_propensity'] for episode in self.trajectories])

        # if self.modeltype == 'conv':
        #     initial_states = env.pos_to_image(env.initial_states())
        # else:
        #     #only works for mountain car
        #     initial_states = np.array([np.tile([x[0],0],self.frameheight).reshape(-1,self.frameheight) for x in env.initial_states()])

        # transitions = np.hstack([ np.vstack([x['x'] for x in self.trajectories]),
        #                           np.hstack([x['a'] for x in self.trajectories]).T.reshape(-1, 1),
        #                           np.hstack([x['r'] for x in self.trajectories]).T.reshape(-1, 1),
        #                           np.vstack([x['x_prime'] for x in self.trajectories]),
        #                           np.hstack([x['done'] for x in self.trajectories]).T.reshape(-1, 1),
        #                           policy_action,
        #                           np.hstack([[n]*len(x['x']) for n,x in enumerate(self.trajectories)]).T.reshape(-1,1),])

        # frames = np.array([x['frames'] for x in self.trajectories])
        # #import pdb; pdb.set_trace()
        print('Training: FQE')
        losses = []
        self.processed_data = self.fill(env)
        self.Q_k_minus_1_all.epoch = 0
        for k in tqdm(range(max_epochs)):
            batch_size = 32

            dataset_length = self.data.num_tuples()
            perm = np.random.permutation(range(dataset_length))
            eighty_percent_of_set = int(1.*len(perm))
            training_idxs = perm[:eighty_percent_of_set]
            validation_idxs = perm[eighty_percent_of_set:]
            training_steps_per_epoch = int(perc_of_dataset * np.ceil(len(training_idxs)/float(batch_size)))
            validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batch_size)))
            # steps_per_epoch = 1 #int(np.ceil(len(dataset)/float(batch_size)))
            train_gen = self.generator(env, pi_e, training_idxs, fixed_permutation=True, batch_size=batch_size)
            # val_gen = self.generator(policy, dataset, validation_idxs, fixed_permutation=True, batch_size=batch_size)

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

    def fill(self, env):
        states = self.data.states()
        states = states.reshape(-1,np.prod(states.shape[2:]))
        actions = self.data.actions().reshape(-1)
        actions = np.eye(env.n_actions)[actions]

        next_states = self.data.next_states()
        original_shape = next_states.shape
        next_states = next_states.reshape(-1,np.prod(next_states.shape[2:]))

        policy_action = self.data.next_target_propensity().reshape(-1, env.n_actions)
        rewards = self.data.rewards().reshape(-1)

        dones = self.data.dones()
        dones = dones.reshape(-1)

        return DataHolder(states, actions, rewards, next_states, dones, policy_action, original_shape)

    @threadsafe_generator
    def generator(self, env, pi_e, all_idxs, fixed_permutation=False,  batch_size = 64):
        """Data Generator for fitting FQE model

        Parameters
        ----------
        env : obj
            The environment object.
        pi_e: obj
            A policy object, evaluation policy.
        all_idxs : ndarray
            1D array of ints representing valid datapoints from which we generate examples
        fixed_permutation : bool, optional
            Run through the data the same way every time?
            Default: False
        batch_size : int
            Minibatch size to during training

        
        Yield
        -------
        obj1, obj2
            obj1: [state, action]
            obj2: [Q]
        """
        # dataset, frames = dataset
        data_length = len(all_idxs)
        steps = int(np.ceil(data_length/float(batch_size)))

        # states = self.data.states()
        # states = states.reshape(-1,np.prod(states.shape[2:]))
        # actions = self.data.actions().reshape(-1)
        # actions = np.eye(env.n_actions)[actions]

        # next_states = self.data.next_states()
        # original_shape = next_states.shape
        # next_states = next_states.reshape(-1,np.prod(next_states.shape[2:]))

        # policy_action = self.data.target_propensity().reshape(-1, env.n_actions)
        # rewards = self.data.rewards().reshape(-1)

        # dones = self.data.dones()
        # dones = dones.reshape(-1)

        states = self.processed_data.states
        actions = self.processed_data.actions
        next_states = self.processed_data.next_states
        original_shape = self.processed_data.original_shape
        policy_action = self.processed_data.policy_action
        rewards = self.processed_data.rewards
        dones = self.processed_data.dones

        alpha = 1.

        # Rebalance dataset
        # probs = np.hstack([np.zeros((dones.shape[0],2)), dones,])[:,:-2]
        # if np.sum(probs):
        #     done_probs = probs / np.sum(probs)
        #     probs = 1 - probs + done_probs
        # else:
        #     probs = 1 - probs
        # probs = probs.reshape(-1)
        # probs /= np.sum(probs)
        # probs = probs[all_idxs]
        # probs /= np.sum(probs)



        # while True:
        #     batch_idxs = np.random.choice(all_idxs, batch_size, p = probs)


        while True:
            perm = np.random.permutation(all_idxs)
            for batch in np.arange(steps):
                batch_idxs = perm[(batch*batch_size):((batch+1)*batch_size)]

                x = states[batch_idxs].reshape(tuple([-1]) + original_shape[2:])
                if self.processor: x = self.processor(x)

                acts = actions[batch_idxs]
                x_ = next_states[batch_idxs].reshape(tuple([-1]) + original_shape[2:])
                if self.processor: x_ = self.processor(x_)

                pi_a_given_x = policy_action[batch_idxs]
                not_dones = 1-dones[batch_idxs]
                rew = rewards[batch_idxs]


                Q_val = self.Q_k_minus_1_all.predict(x_).reshape(pi_a_given_x.shape)
                # if self.Q_k_minus_1_all.epoch == 0:
                #     Q_val = np.zeros_like(Q_val)
                # Q_val = Q_val[np.arange(len(acts)), np.argmax(acts,axis=1)]
                Q_val = (Q_val * pi_a_given_x).sum(axis=-1)
                new_Q = rew + self.gamma * (Q_val * not_dones).reshape(-1)

                old_Q = 0 #(self.Q_k.predict([x, acts]).reshape(-1) * not_dones)
                Q = (old_Q) + (alpha)*(new_Q-old_Q) # Q-learning style update w/ learning rate, to stabilize

                yield ([x, acts], Q)
