from ope.algos.direct_method import DirectMethodQ
import sys
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from ope.utls.thread_safe import threadsafe_generator
from keras import regularizers
from sklearn.linear_model import LinearRegression, LogisticRegression

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

class MRDR(DirectMethodQ):
    """Algorithm: More Robust Doubly Robust (MRDR).
    """
    def __init__(self) -> None:
        DirectMethodQ.__init__(self)
    
    def fit_NN(self, data, pi_e, config, verbose=True) -> float:
        """(Neural) Get the MRDR OPE estimate for pi_e.

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
        batch_size: int
            Batch size for the quad program
        epsilon : float, optional
            Default: 0.001
        
        Returns
        -------
        float, obj1, obj2
            float: represents the average value of the final 10 iterations
            obj1: Fitted Forward model Q(s,a) -> R
            obj2: Fitted Forward model Q(s) -> R^|A| 
        """

        cfg = config.models['MRDR']
        processor = config.processor
        
        # TODO: early stopping + lr reduction
        
        im = data.states()[0]
        if processor: im = processor(im)
        self.Q_k = cfg['model'](im.shape[1:], data.n_actions)
        optimizer = optim.Adam(self.Q_k.parameters())

        print('Training: MRDR')
        losses = []
        for k in tqdm(range(cfg['max_epochs'])):
            batch_size = cfg['batch_size']
            
            dataset_length = data.num_tuples()
            perm = np.random.permutation(range(dataset_length))

            perm = np.random.permutation(data.idxs_of_non_abs_state())

            eighty_percent_of_set = int(.8*len(perm))
            training_idxs = perm[:eighty_percent_of_set]
            validation_idxs = perm[eighty_percent_of_set:]
            training_steps_per_epoch = int(1. * np.ceil(len(training_idxs)/float(batch_size)))
            validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batch_size)))
            # steps_per_epoch = 1 #int(np.ceil(len(dataset)/float(batch_size)))
            train_gen = self.generator(data, config, training_idxs, fixed_permutation=True, batch_size=batch_size, processor=processor)
            val_gen = self.generator(data, config, training_idxs, fixed_permutation=True, batch_size=batch_size, processor=processor)

            M = 5
        
            for step in range(training_steps_per_epoch):
                
                with torch.no_grad():
                    inp, _ = next(train_gen)
                    states = torch.from_numpy(inp[0]).float()
                    actions = torch.from_numpy(inp[1]).bool()
                    weights = torch.from_numpy(inp[2]).float()
                    rew = torch.from_numpy(inp[3]).float()
                    behavior_propensity = torch.from_numpy(inp[4]).float()
                    target_propensity = torch.from_numpy(inp[5]).float()

                prediction = self.Q_k.predict(states)

                Omega = torch.diag_embed(behavior_propensity.pow(-1)) - 1
                D = torch.diag_embed(target_propensity)
                qbeta = torch.matmul(D, torch.unsqueeze(prediction, 2)) - torch.unsqueeze(rew, 2)
                qbeta_T = torch.transpose(qbeta, 2, 1)
                unweighted_loss = torch.matmul(torch.matmul(qbeta_T, Omega), qbeta)
                weighted_loss = weights.view((-1, 1)) * unweighted_loss.view((-1, 1))
                loss  = weighted_loss.squeeze().mean()

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.Q_k.parameters(), cfg['clipnorm'])
                optimizer.step()

        self.fitted = 'NN'
        return 1.0

    @threadsafe_generator
    def generator(self, data, cfg, all_idxs, fixed_permutation=False,  batch_size = 64, processor=None):
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
        is_train : bool
            Deprecated 

        
        Yield
        -------
        obj1, obj2
            obj1: [state, actions, weights, rewards, base propensities, target propensities]
            obj2: []
        """
        # dataset, frames = dataset
        data_length = len(all_idxs)
        steps = int(np.ceil(data_length/float(batch_size)))

        n = len(data)
        T = max(data.lengths())
        n_dim = data.n_dim
        n_actions = data.n_actions

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
        omega = [np.cumprod(om) for om in data.omega()]
        gamma_vec = cfg.gamma**np.arange(T)
        actions = data.actions()
        rewards = data.rewards()

        factors, Rs = [], []
        for traj_num, ts in tqdm(enumerate(data.ts())):
            for t in ts:
                i,t = int(traj_num), int(t)
                R = np.zeros(n_actions)
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

        states = data.states()
        original_shape = states.shape
        states = states.reshape(-1,np.prod(states.shape[2:]))

        actions = np.eye(n_actions)[actions.reshape(-1)]

        base_propensity = data.base_propensity().reshape(-1, n_actions)
        target_propensity = data.target_propensity().reshape(-1, n_actions)

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

                if processor: x = processor(x)

                yield ([x, acts, weights, rs, pib, pie], [])

    def Q_NN(self, states, actions=None) -> np.ndarray:
        if actions is None:
            return self.Q_k.predict(torch.from_numpy(states).float()).detach().numpy()
        else:
            return self.Q_k.predict(torch.from_numpy(states).float())[np.arange(len(actions)), actions].detach().numpy()
    
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
        pi_e: obj
            A policy object, evaluation policy.
        
        Returns
        -------
        ndarray
            2D array containing data with float type. 
            Each row represents the feature phi(state, action)
        """

        n = len(data)
        T = max(data.lengths())
        n_dim = data.n_dim
        n_actions = data.n_actions

        data_dim = n * T * n_actions

        phi = data_dim * [None]
        target_propensity = data.target_propensity()

        for i in range(n):
            states = np.squeeze(data.states(low_=i, high_=i+1))
            l = data.lengths()[i]
            for t in range(T):
                for action in range(n_actions):
                    if t < l:
                        s = states[t]
                        pie_s_t_a_t = target_propensity[i][t][action] #pi_e.predict([states[t]])[0][action]
                        phi[(i * T + t) * n_actions + action] = pie_s_t_a_t * self.compute_feature(s, action, t, n_dim, n_actions)
                    else:
                        phi[(i * T + t) * n_actions + action] = np.zeros(len(phi[0]))


        return np.array(phi, dtype='float')

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
        """(Tabular) Get the MRDR OPE Q function for pi_e.

        Parameters
        ----------
        pi_e: obj
            A policy object, evaluation policy.
        
        Returns
        -------
        obj
            obj: Returns itself equipped with a predict function
        """
        cfg = config.models['MRDR']
        epsilon = cfg['convergence_epsilon']
        max_epochs = cfg['max_epochs']
        gamma = config.gamma

        full_dataset = data

        n = len(data)
        T = max(data.lengths())
        n_dim = data.n_dim
        n_actions = data.n_actions
        rho = [np.cumprod(om) for om in data.omega()]
        gamma_vec = gamma**np.arange(T)

        data_dim = n * n_actions * T
        omega = n*T * [None]
        r_tild = np.zeros(data_dim)
        for i in tqdm(range(n)):


            states = np.squeeze(data.states(low_=i, high_=i+1))
            actions = data.actions()[i]
            rewards = data.rewards()[i]
            pi0 = data.base_propensity()[i]
            pi1 = data.target_propensity()[i]
            l = data.lengths()[i]



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
                    rho_prev =  rho[i][t-1]

                if rho_prev:
                    propensity_weight_t = gamma_vec[t] ** 2 * rho_prev ** 2 * (rho[i][t] / rho_prev)
                else:
                    propensity_weight_t = 0

                om = propensity_weight_t * D_pi_e.dot(omega_s_t).dot(D_pi_e)
                omega[i * T + t] = om

                t_limit = min(T, l)
                if rho[i][t]:
                    val = np.sum((rho[i][t:t_limit] / rho[i][t]) * (gamma_vec[t:t_limit] / gamma_vec[t]) * rewards[t:])
                else:
                    val = 0
                r_tild[(i * T + t) * n_actions + action_t] = val

        alpha = 1
        lamb = 1
        cond_number_threshold_A = 10000
        block_size = int(n_actions * n * T/4)

        phi = self.compute_grid_features(full_dataset)
        self.weight = self.wls_sherman_morrison(phi, r_tild, omega, lamb, alpha, cond_number_threshold_A, block_size)

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
