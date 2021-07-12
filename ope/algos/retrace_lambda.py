from ope.algos.direct_method import DirectMethodQ
import sys
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from ope.utls.thread_safe import threadsafe_generator
from sklearn.linear_model import LinearRegression, LogisticRegression

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

class Retrace(DirectMethodQ):
    """Algorithm: Retrace Family (Retrace(lambda), Q(lambda), Tree-Backup(lambda)).
    """
    def __init__(self, method, lamb) -> None:
        DirectMethodQ.__init__(self)
        self.method = method
        self.lamb = lamb

    def fit_tabular(self, data, pi_e, config, verbose = True):
        """(Tabular) Get the Retrace/Tree/Q^pi OPE Q function for pi_e.

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
        lamb = self.lamb
        method = self.method
        cfg = config.models[self.method]
        epsilon = cfg['convergence_epsilon']
        divergence_epsilon = cfg['divergence_epsilon']
        max_epochs = cfg['max_epochs']
        gamma = config.gamma
        
        S = np.squeeze(data.states())
        SN = np.squeeze(data.next_states())
        ACTS = data.actions()
        REW = data.rewards()
        PIE = data.target_propensity()
        PIB = data.base_propensity()
        DONES = data.dones()

        unique_states = np.unique(np.vstack([S, SN]))
        state_space_dim = len(unique_states)
        action_space_dim = data.n_actions

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

                if method == 'Retrace':
                    c = pie[range(len(a)), a]/(pib[range(len(a)), a] + eps)
                    c[0] = 1.
                    c = lamb * np.minimum(1., c) # c_s = lambda * min(1, pie/pib)
                elif method == 'Tree-Backup':
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
                gam = gamma ** np.arange(t)

                expected_U = np.sum(pi_e.predict(sn)*U[sn, :], axis=1)*(1-done)
                # expected_U = np.sum([], axu

                diff = r + gamma * expected_U - U[s, a]

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
            if delta < epsilon or count > max_epochs or delta > divergence_epsilon:# * (1 - self.gamma) / self.gamma:
                # return np.sum([prob*U1[0, new_a] for new_a,prob in enumerate(pi_e.predict([0])[0])]), U1, mapping #U[0,pi_e([0])][0]
                self.table = U1
                self.mapping = mapping
                break
        self.fitted = 'tabular'

    def Q_tabular(self, states, actions=None) -> np.ndarray:
        if actions is None:
            return np.array([self.table[self.mapping[state]] for state in np.squeeze(states)])
        else:
            return np.array([self.table[self.mapping[state]] for state in np.squeeze(states)])[np.arange(len(actions)), actions]

    def Q_NN(self, states, actions=None) -> np.ndarray:
        if actions is None:
            return self.Q_k.predict(torch.from_numpy(states).float()).detach().numpy()
        else:
            return self.Q_k.predict(torch.from_numpy(states).float())[np.arange(len(actions)), actions].detach().numpy()
    
    @staticmethod
    def copy_over_to(source, target):
        target.load_state_dict(source.state_dict())

    def fit_NN(self, data, pi_e, config, verbose=True) -> float:
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
        cfg = config.models[self.method]
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

        print('Training: %s' % self.method)
        losses = []
        
        batch_size = cfg['batch_size']
        dataset_length = data.num_tuples()
        perm = np.random.permutation(range(dataset_length))
        eighty_percent_of_set = int(1.*len(perm))
        training_idxs = perm[:eighty_percent_of_set]
        validation_idxs = perm[eighty_percent_of_set:]
        training_steps_per_epoch = max(500, int(.03 * np.ceil(len(training_idxs)/float(batch_size))))
        validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batch_size)))
        # steps_per_epoch = 1 #int(np.ceil(len(dataset)/float(batch_size)))
        
        for k in tqdm(range(cfg['max_epochs'])):
            train_gen = self.generator(data, config, training_idxs, fixed_permutation=True, batch_size=batch_size, processor=processor)
            # val_gen = self.generator(policy, dataset, validation_idxs, method, fixed_permutation=True, batch_size=batch_size)

            # import pdb; pdb.set_trace()
            # train_gen = self.generator(env, pi_e, (transitions,frames), training_idxs, fixed_permutation=True, batch_size=batch_size)
            # inp, out = next(train_gen)

            M = 5
        
            for step in range(training_steps_per_epoch):
                
                with torch.no_grad():
                    inp, out = next(train_gen)
                    states = torch.from_numpy(inp[0]).float()
                    actions = torch.from_numpy(inp[1]).bool()
                    output = torch.from_numpy(out[0]).float()

                prediction = self.Q_k(states, actions)
                loss = (prediction - output).pow(2).mean()
                
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.Q_k.parameters(), cfg['clipnorm'])
                optimizer.step()
                

            self.copy_over_to(self.Q_k, self.Q_k_minus_1)

            actions = pi_e.sample(initial_states)
            assert len(actions) == initial_states.shape[0]
            Q_val = self.Q_k.predict(torch.from_numpy(initial_states).float())[np.arange(len(actions)), actions].detach().numpy()
            values.append(np.mean(Q_val))
            print(values[-1], np.mean(values[-M:]), np.abs(np.mean(values[-M:])- np.mean(values[-(M+1):-1])), 1e-4*np.abs(np.mean(values[-(M+1):-1])))
            if k>M and np.abs(np.mean(values[-M:]) - np.mean(values[-(M+1):-1])) < 1e-4*np.abs(np.mean(values[-(M+1):-1])):
                break
        
        self.fitted = 'NN'
        return np.mean(values[-10:])

    @threadsafe_generator
    def generator(self, data, cfg, all_idxs, fixed_permutation=False,  batch_size = 64, processor=None):
        """Data Generator for fitting Retrace model

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

        states = data.states()
        # states = states.reshape(-1,np.prod(states.shape[2:]))
        actions = data.actions()
        # actions = np.eye(env.n_actions)[actions]

        next_states = data.next_states()
        original_shape = next_states.shape
        next_states = next_states.reshape(-1,np.prod(next_states.shape[2:]))

        pi1_ = data.next_target_propensity()
        pi1 = data.target_propensity()
        pi0 = data.base_propensity()
        rewards = data.rewards()

        dones = data.dones()
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

                traj_num = int(idx/ data.lengths()[0]) # Assume fixed length, horizon is fixed
                i = idx - traj_num * data.lengths()[0]
                s = data.states(low_=traj_num, high_=traj_num+1)[0,i:]
                sn = data.next_states(low_=traj_num, high_=traj_num+1)[0,i:]
                a = actions[traj_num][i:]
                r = rewards[traj_num][i:]
                pie = pi1[traj_num][i:]
                pie_ = pi1_[traj_num][i:]
                pib = pi0[traj_num][i:]

                if self.method == 'Retrace':
                    c = pie[range(len(a)), a]/pib[range(len(a)), a]
                    c[0] = 1.
                    c = self.lamb * np.minimum(1., c) # c_s = lambda * min(1, pie/pib)
                elif self.method == 'Tree-Backup':
                    c = self.lamb * pie[range(len(a)), a] # c_s = lambda * pi(a|x)
                    c[0] = 1.
                elif self.method == 'Q^pi(lambda)':
                    c = np.ones_like(a)*self.lamb # c_s = lambda
                    c[0] = 1.
                elif self.method == 'IS':
                    c = pie[range(len(a)), a]/pib[range(len(a)), a] # c_s = pie/pib
                    c[0] = 1.
                    c = c # c_s = pie/pib
                else:
                    raise

                c = np.cumprod(c)
                gam = cfg.gamma ** np.arange(len(s))

                if processor:
                    s = processor(s)
                    sn = processor(sn)

                Q_x = self.Q_k_minus_1.predict(torch.from_numpy(s).float()).detach().numpy()
                Q_x_ = self.Q_k_minus_1.predict(torch.from_numpy(sn).float()).detach().numpy()

                Q_xt_at = Q_x[range(len(a)), a]
                E_Q_x_ = np.sum(Q_x_*pie_, axis=1)

                Ss.append(s[0])
                As.append(a[0])
                Ys.append(Q_xt_at[0] + np.sum(gam * c * (r + cfg.gamma * E_Q_x_ - Q_xt_at)))

            yield [np.array(Ss), np.eye(data.n_actions)[np.array(As)]], [np.array(Ys)]
