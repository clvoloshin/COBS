from ope.algos.direct_method import DirectMethod
import sys
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from ope.utls.thread_safe import threadsafe_generator
from sklearn.linear_model import LinearRegression, LogisticRegression
from collections import Counter

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

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

class FittedQEvaluation(DirectMethod):
    """Algorithm: Fitted Q Evaluation (FQE).
    """
    def __init__(self) -> None:
        DirectMethod.__init__(self)

    def fit_tabular(self, data, pi_e, config, verbose = True):
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
        cfg = config.models['FQE']
        epsilon = cfg['convergence_epsilon']
        max_epochs = cfg['max_epochs']
        gamma = config.gamma

        action_space_dim = data.n_actions
        data = data.basic_transitions()
        
        state_space_dim = len(np.unique(data[:,[0, 3]].reshape(-1)))
        # L = max(data[:,-1]) + 1

        mapping = {state:idx for idx,state in enumerate(np.unique(data[:,[0,3]].reshape(-1)))}

        U1 = np.zeros(shape=(state_space_dim, action_space_dim))
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
                vals = np.array(group['r']) + gamma * np.sum(pi_e.predict(x_primes)*U[x_primes, :], axis=1)*(1-np.array(group['done']))
                # for act in range(action_space_dim):
                #     try:

                #         vals += self.gamma*pi_e.predict(np.array(group['x_prime']))[range(len(x_primes)), act ]*U[x_primes,act]*(1-group['done'])
                #     except:
                #         import pdb; pdb.set_trace()

                # vals += group['r']

                U1[x, a] = np.mean(vals)#expected_reward + self.gamma*expected_Q

                delta = max(delta, abs(U1[x,a] - U[x,a]))
            
            if verbose: print(count, delta)

            if gamma == 1:
                # TODO: include initial state distribution
                if delta < epsilon:
                    out = np.sum([prob*U1[0, new_a] for new_a,prob in enumerate(pi_e.predict([0])[0])]) #U[0,pi_e([0])][0]
                    # return None, U1, mapping
                    self.table = U1
                    self.mapping = mapping
                    break
                    # return out, U1, mapping
            else:
                if delta < epsilon * (1 - gamma) / gamma or count > max_epochs:
                    # return None, U1, mapping #U[0,pi_e([0])][0]
                    self.table = U1
                    self.mapping = mapping
                    break
                    # return np.sum([prob*U1[mapping[0], new_a] for new_a,prob in enumerate(pi_e.predict([0])[0])]), U1, mapping #U[0,pi_e([0])][0]
        
        self.fitted = 'tabular'
    
    def Q_tabular(self, states, actions=None) -> np.ndarray:
        if actions is None:
            return self.table[self.mapping[states]]
        else:
            return self.table[self.mapping[states]][np.arange(len(actions)), actions]

    def Q_NN(self, states, actions=None) -> np.ndarray:
        if actions is None:
            return self.Q_k.predict(torch.from_numpy(states).float()).detach().numpy()
        else:
            return self.Q_k.predict(torch.from_numpy(states).float())[np.arange(len(actions)), actions].detach().numpy()
    
    @staticmethod
    def copy_over_to(source, target):
        target.load_state_dict(source.state_dict())

    def fit_NN(self, data, pi_e, config, verbose=True) -> float:
        cfg = config.models['FQE']
        processor = config.processor

        initial_states = data.initial_states()
        if processor: initial_states = processor(initial_states)
        
        im = data.states()[0]
        if processor: im = processor(im)
        self.Q_k = cfg['model'](im.shape[1:], data.n_actions)
        self.Q_k.apply(cfg['model'].weight_init) # weight initializer 
        self.Q_k_minus_1 = cfg['model'](im.shape[1:], data.n_actions)
        optimizer = optim.Adam(self.Q_k.parameters())
        
        self.copy_over_to(self.Q_k, self.Q_k_minus_1)
        values = []

        print('Training: FQE')
        processed_data = self.fill(data)
        losses = []
        for k in tqdm(range(cfg['max_epochs'])):
            batch_size = cfg['batch_size']

            dataset_length = data.num_tuples()
            perm = np.random.permutation(range(dataset_length))
            eighty_percent_of_set = int(1.*len(perm))
            training_idxs = perm[:eighty_percent_of_set]
            validation_idxs = perm[eighty_percent_of_set:]
            training_steps_per_epoch = int(1. * np.ceil(len(training_idxs)/float(batch_size)))
            train_gen = self.generator(processed_data, config, training_idxs, fixed_permutation=True, batch_size=batch_size, processor=processor)
            
            M = 5
        
            for step in range(training_steps_per_epoch):
                
                with torch.no_grad():
                    inp, out = next(train_gen)
                    states = torch.from_numpy(inp[0]).float()
                    actions = torch.from_numpy(inp[1]).bool()
                    output = torch.from_numpy(out).float()

                prediction = self.Q_k(states, actions)
                loss = (prediction - output).pow(2).mean()
                
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.Q_k.parameters(), cfg['clipnorm'])
                optimizer.step()
                

            self.copy_over_to(self.Q_k, self.Q_k_minus_1)

            # losses.append(hist.history['loss'])
            actions = pi_e.sample(initial_states)
            assert len(actions) == initial_states.shape[0]
            
            Q_val = self.Q_k.predict(torch.from_numpy(initial_states).float())[np.arange(len(actions)), actions].detach().numpy()
            values.append(np.mean(Q_val))
            if verbose: print(values[-1], np.mean(values[-M:]), np.abs(np.mean(values[-M:])- np.mean(values[-(M+1):-1])), 1e-4*np.abs(np.mean(values[-(M+1):-1])))
            if k>M and np.abs(np.mean(values[-M:]) - np.mean(values[-(M+1):-1])) < 1e-4*np.abs(np.mean(values[-(M+1):-1])):
                break

        self.fitted = 'NN'
        return np.mean(values[-10:])
    
    def fill(self, data):
        states = data.states()
        states = states.reshape(-1,np.prod(states.shape[2:]))
        actions = data.actions().reshape(-1)
        actions = np.eye(data.n_actions)[actions]

        next_states = data.next_states()
        original_shape = next_states.shape
        next_states = next_states.reshape(-1,np.prod(next_states.shape[2:]))

        policy_action = data.next_target_propensity().reshape(-1, data.n_actions)
        rewards = data.rewards().reshape(-1)

        dones = data.dones()
        dones = dones.reshape(-1)

        return DataHolder(states, actions, rewards, next_states, dones, policy_action, original_shape)
    
    @threadsafe_generator
    def generator(self, data, cfg, all_idxs, fixed_permutation=False,  batch_size = 64, processor=None):
        """Data Generator for fitting FQE model

        Parameters
        ----------
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

        states = data.states
        actions = data.actions
        next_states = data.next_states
        original_shape = data.original_shape
        policy_action = data.policy_action
        rewards = data.rewards
        dones = data.dones

        alpha = 1.

        while True:
            perm = np.random.permutation(all_idxs)
            for batch in np.arange(steps):
                batch_idxs = perm[(batch*batch_size):((batch+1)*batch_size)]

                x = states[batch_idxs].reshape(tuple([-1]) + original_shape[2:])
                if processor: x = processor(x)

                acts = actions[batch_idxs]
                x_ = next_states[batch_idxs].reshape(tuple([-1]) + original_shape[2:])
                if processor: x_ = processor(x_)

                pi_a_given_x = policy_action[batch_idxs]
                not_dones = 1-dones[batch_idxs]
                rew = rewards[batch_idxs]

                Q_val = self.Q_k_minus_1.predict(torch.from_numpy(x_).float()).reshape(pi_a_given_x.shape).detach().numpy()
                # if self.Q_k_minus_1_all.epoch == 0:
                #     Q_val = np.zeros_like(Q_val)
                # Q_val = Q_val[np.arange(len(acts)), np.argmax(acts,axis=1)]
                Q_val = (Q_val * pi_a_given_x).sum(axis=-1)
                new_Q = rew + cfg.gamma * (Q_val * not_dones).reshape(-1)

                old_Q = 0 #(self.Q_k.predict([x, acts]).reshape(-1) * not_dones)
                Q = (old_Q) + (alpha)*(new_Q-old_Q) # Q-learning style update w/ learning rate, to stabilize

                yield ([x, acts], Q)
        
    # def run_linear(self, env, pi_b, pi_e, max_epochs, epsilon=.001, fit_intercept=True):
    #     """(Linear) Get the FQE OPE estimate.

    #     Parameters
    #     ----------
    #     env : obj
    #         The environment object.
    #     pi_b : obj
    #         A policy object, behavior policy.
    #     pi_e: obj
    #         A policy object, evaluation policy.
    #     max_epochs : int
    #         Max number of iterations
    #     epsilon : float
    #         Convergence criteria.
    #         Default: 0.001
    #     fit_intercept : bool
    #         Fit the y-intercept
    #         Default: True
        
    #     Returns
    #     -------
    #     obj
    #         sklearn LinearReg object representing the Q function
    #     """
    #     initial_states = self.data.initial_states()
    #     self.Q_k = LinearRegression(fit_intercept=fit_intercept)
    #     values = []

    #     states = self.data.states()
    #     states = states.reshape(-1,np.prod(states.shape[2:]))
    #     actions = self.data.actions().reshape(-1)
    #     actions = np.eye(env.n_actions)[actions]
    #     X = np.hstack([states, actions])

    #     next_states = self.data.next_states()
    #     next_states = next_states.reshape(-1,np.prod(next_states.shape[2:]))

    #     policy_action = self.data.target_propensity()
    #     lengths = self.data.lengths()
    #     omega = self.data.omega()
    #     rewards = self.data.rewards()

    #     not_dones = 1-self.data.dones()

    #     for epoch in tqdm(range(max_epochs)):

    #         if epoch:
    #             inp = np.repeat(next_states, env.n_actions, axis=0)
    #             act = np.tile(np.arange(env.n_actions), len(next_states))
    #             inp = np.hstack([inp.reshape(inp.shape[0],-1), np.eye(env.n_actions)[act]])
    #             Q_val = self.Q_k.predict(inp).reshape(policy_action.shape)
    #         else:
    #             Q_val = np.zeros_like(policy_action)
    #         Q = rewards + self.gamma * (Q_val * policy_action).sum(axis=-1) * not_dones
    #         Q = Q.reshape(-1)

    #         self.Q_k.fit(X, Q)

    #         # Check if converged
    #         actions = pi_e.sample(initial_states)
    #         Q_val = self.Q_k.predict(np.hstack([initial_states.reshape(initial_states.shape[0],-1), np.eye(env.n_actions)[actions]]))
    #         values.append(np.mean(Q_val))
    #         M = 20
    #         # print(values[-1], np.mean(values[-M:]), np.abs(np.mean(values[-M:])- np.mean(values[-(M+1):-1])), 1e-4*np.abs(np.mean(values[-(M+1):-1])))
    #         if epoch>M and np.abs(np.mean(values[-M:]) - np.mean(values[-(M+1):-1])) < 1e-4*np.abs(np.mean(values[-(M+1):-1])):
    #             break
    #     #np.mean(values[-10:]), self.Q_k,
    #     return self.Q_k

    # def run_linear_value_iter(self, env, pi_b, pi_e, max_epochs, epsilon=.001):
    #     initial_states = self.data.initial_states()
    #     self.Q_k = LinearRegression()
    #     values = []

    #     states = self.data.states()
    #     states = states.reshape(-1,np.prod(states.shape[2:]))
    #     actions = self.data.actions().reshape(-1)
    #     actions = np.eye(env.n_actions)[actions]
    #     X = states #np.hstack([states, actions])

    #     next_states = self.data.next_states()
    #     next_states = next_states.reshape(-1,np.prod(next_states.shape[2:]))

    #     policy_action = self.data.target_propensity()
    #     lengths = self.data.lengths()
    #     omega = self.data.omega()
    #     rewards = self.data.rewards()

    #     not_dones = 1-self.data.dones()

    #     for epoch in tqdm(range(max_epochs)):


    #         if epoch:
    #             # inp = np.repeat(next_states, env.n_actions, axis=0)
    #             inp = next_states
    #             # act = np.tile(np.arange(env.n_actions), len(next_states))
    #             inp = inp.reshape(inp.shape[0],-1) #np.hstack([inp.reshape(inp.shape[0],-1), np.eye(env.n_actions)[act]])
    #             Q_val = self.Q_k.predict(inp).reshape(policy_action[...,0].shape)
    #         else:
    #             Q_val = np.zeros_like(policy_action[...,0]) + 1
    #         Q = rewards + self.gamma * Q_val * not_dones
    #         Q = Q.reshape(-1)

    #         self.Q_k.fit(X, Q)

    #         # Check if converged
    #         actions = pi_e.sample(initial_states)
    #         # Q_val = self.Q_k.predict(np.hstack([initial_states.reshape(initial_states.shape[0],-1), np.eye(env.n_actions)[actions]]))
    #         Q_val = self.Q_k.predict(initial_states.reshape(initial_states.shape[0],-1)) #self.Q_k.predict(np.hstack([, np.eye(env.n_actions)[actions]]))
    #         values.append(np.mean(Q_val))
    #         M = 20
    #         # print(values[-1], np.mean(values[-M:]), np.abs(np.mean(values[-M:])- np.mean(values[-(M+1):-1])), 1e-4*np.abs(np.mean(values[-(M+1):-1])))
    #         print(self.Q_k.coef_)
    #         if epoch>M and np.abs(np.mean(values[-M:]) - np.mean(values[-(M+1):-1])) < 1e-4*np.abs(np.mean(values[-(M+1):-1])):
    #             break

    #     #np.mean(values[-10:]), self.Q_k,
    #     return self.Q_k

