
from ope.algos.direct_method import DirectMethodQ
import sys
import numpy as np
import pandas as pd
sys.path.append("..")
from copy import deepcopy
from tqdm import tqdm
from ope.utls.thread_safe import threadsafe_generator
from sklearn.linear_model import LinearRegression, LogisticRegression
from functools import partial

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)


class DirectMethodRegression(DirectMethodQ):
    """Algorithm: Direct Model Regression (Q-Reg).
    """
    def __init__(self) -> None:
        DirectMethodQ.__init__(self)

    def wls_sherman_morrison(self, phi_in, rewards_in, omega_in, lamb, omega_regularizer, cond_number_threshold_A, block_size=None):
        """Weighted Least Squares via Sherman Morrison Algorithm.

        Parameters
        ----------
        phi_in : ndarray
            2D array containing data with float type. 
            Feature vector phi(state, action).
        rewards_in : list
            1D list containing data with float type. 
            Discounted reward to go.
        omega_in : list
            1D list containing data with float type. 
            Distribution correction factors.
        lamb : float
            Deprecated
        omega_regularizer : float
            Additive regularization for correction factors.
        cond_number_threshold_A : float
            Deprecated.
        block_size : int, optional
            Deprecated.
        
        Returns
        -------
        list
            Weights representing the linear model weights^T * features.
        """
        
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

    def fit_tabular(self, data, pi_e, config, verbose = True):
        """(Tabular) Get the Q-Reg OPE Q function for pi_e via weighted least squares.

        Parameters
        ----------
        pi_b : obj
            A policy object, behavior policy.
        pi_e: obj
            A policy object, evaluation policy.
        epsilon : float
            Deprecated.
        
        Returns
        -------
        obj, DMModel
            An object representing the Q function
        """
        cfg = config.models['Q-Reg']
        epsilon = cfg['convergence_epsilon']
        max_epochs = cfg['max_epochs']
        gamma = config.gamma

        full_dataset = data
        dataset = data.all_transitions()
        frames = data.frames()
        omega = data.omega()
        rewards = data.rewards()

        omega = [np.cumprod(om) for om in omega]
        gamma_vec = gamma**np.arange(max([len(x) for x in omega]))

        factors, Rs = [], []
        for data in dataset:
            ts = data[-1]
            traj_num = data[-2]

            i,t = int(traj_num), int(ts)
            Rs.append( np.sum( omega[i][t:]/omega[i][t] * gamma_vec[t:]/gamma_vec[t] *  rewards[i][t:] )  )
            factors.append( gamma_vec[t] * omega[i][t] )

        alpha = 1
        lamb = 1
        cond_number_threshold_A = 1
        block_size = len(dataset)

        phi = self.compute_grid_features(full_dataset)
        self.weight = self.wls_sherman_morrison(phi, Rs, factors, lamb, alpha, cond_number_threshold_A, block_size)

        
        self.n_dim = full_dataset.n_dim
        self.n_actions = full_dataset.n_actions

        self.fitted = 'tabular'

    def Q_tabular(self, states, actions=None) -> np.ndarray:
        if actions is None:
            Q = np.array([[np.matmul(self.weight, self.compute_feature(s, a, 0, self.n_dim, self.n_actions)) for a in range(self.n_actions)] for s in np.squeeze(states)])
            return Q
        else:
            #TODO
            NotImplemented
            # import pdb; pdb.set_trace()
            # Q = []
            # for (s,a) in zip(states, actions):
            #     Q.append(np.matmul(self.weight, self.compute_feature(s, a, 0, self.n_dim, self.n_actions)))
            # return np.array(Q)

    def compute_feature(self, state, action, step, n_dim, n_actions):
        """Feature map. One hot encoding of state-action.

        Parameters
        ----------
        state : int
            State.
        action: int
            Action.
        step : int
            Deprecated.
        
        Returns
        -------
        list
            One hot encoding of the state-action that was taken.
            phi[state, action] = 1 otherwise 0.
        """

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

    def compute_grid_features(self, data):
        """Get features for the dataset.

        Parameters
        ----------
        None
        
        Returns
        -------
        ndarray
            2D array containing data with float type. 
            Each row represents the feature phi(state, action)
        """

        T = max(data.lengths())
        n_dim = data.n_dim
        n_actions = data.n_actions


        n = len(data)


        data_dim = n * T

        phi = data_dim * [None]

        lengths = data.lengths()
        for i in range(n):
            states = data.states(False, i, i+1)
            actions = data.actions()[i]

            for t in range(max(lengths)):
                if t < lengths[i]:
                    s = states[t]
                    action = int(actions[t])
                    phi[i * T + t] = self.compute_feature(s, action, t, n_dim, n_actions)
                else:
                    phi[i * T + t] = np.zeros(len(phi[0]))

        return np.array(phi, dtype='float')

    def Q_NN(self, states, actions=None) -> np.ndarray:
        if actions is None:
            return self.Q_k.predict(torch.from_numpy(states).float()).detach().numpy()
        else:
            return self.Q_k.predict(torch.from_numpy(states).float())[np.arange(len(actions)), actions].detach().numpy()
    
    def fit_NN(self, data, pi_e, config, verbose=True) -> float:
        """(Neural) Get the Q-Reg OPE estimate for pi_e via weighted least squares.

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
        
        Returns
        -------
        obj1, obj2
            obj1: Fitted Forward model Q(s,a) -> R
            obj2: Fitted Forward model Q(s) -> R^|A| 
        """
        cfg = config.models['Q-Reg']
        processor = config.processor
        
        # TODO: early stopping + lr reduction
        
        im = data.states()[0]
        if processor: im = processor(im)
        self.Q_k = cfg['model'](im.shape[1:], data.n_actions)
        optimizer = optim.Adam(self.Q_k.parameters())
        
        print('Training: Model Free')
        losses = []
        for k in tqdm(range(cfg['max_epochs'])):
            batch_size = cfg['batch_size']

            dataset_length = data.num_tuples()
            perm = np.random.permutation(range(dataset_length))
            eighty_percent_of_set = int(.8*len(perm))
            training_idxs = perm[:eighty_percent_of_set]
            validation_idxs = perm[eighty_percent_of_set:]
            training_steps_per_epoch = int(1.*np.ceil(len(training_idxs)/float(batch_size)))
            validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batch_size)))
            train_gen = self.generator(data, config, training_idxs, fixed_permutation=True, batch_size=batch_size, processor=processor)
            val_gen = self.generator(data, config, training_idxs, fixed_permutation=True, batch_size=batch_size, processor=processor)

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

        self.fitted = 'NN'
        return 1.0

    @threadsafe_generator
    def generator(self, data, cfg, all_idxs, fixed_permutation=False,  batch_size = 64, processor=None):
        """Data Generator for fitting Q-Reg model

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
            obj1: [state, action, weight]
            obj2: [reward]
        """
        data_length = len(all_idxs)
        steps = int(np.ceil(data_length/float(batch_size)))

        states = data.states()
        states = states.reshape(tuple([-1]) + states.shape[2:])
        lengths = data.lengths()
        omega = data.omega()
        rewards = data.rewards()
        actions = data.actions().reshape(-1)

        omega = [np.cumprod(om) for om in omega]
        gamma_vec = cfg.gamma**np.arange(max([len(x) for x in omega]))

        factors, Rs = [], []
        for traj_num, ts in enumerate(data.ts()):
            for t in ts:
                i,t = int(traj_num), int(t)
                if omega[i][t]:
                    Rs.append( np.sum( omega[i][t:]/omega[i][t] * gamma_vec[t:]/gamma_vec[t] *  rewards[i][t:] )  )
                else:
                    Rs.append( 0 )
                factors.append( gamma_vec[t] * omega[i][t] )

        Rs = np.array(Rs)
        factors = np.array(factors)

        dones = data.dones()
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

        while True:
            perm = np.random.permutation(all_idxs)
            for batch in np.arange(steps):
                batch_idxs = perm[(batch*batch_size):((batch+1)*batch_size)]

                x = states[batch_idxs]
                if processor: x = processor(x)
                weight = factors[batch_idxs] #* probs[batch_idxs]
                R = Rs[batch_idxs]
                acts = actions[batch_idxs]

                yield ([x, np.eye(data.n_actions)[acts], np.array(weight).reshape(-1,1)], [np.array(R).reshape(-1,1)])
