from ope.algos.direct_method import DirectMethodWeight
import numpy as np
from time import sleep
import sys
import os
from tqdm import tqdm
import json

from scipy.optimize import linprog
from scipy.optimize import minimize
import quadprog

from tqdm import tqdm

from ope.utls.thread_safe import threadsafe_generator

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)


class InfiniteHorizonOPE(DirectMethodWeight):
    """Algorithm: Infinite Horizon (IH).
    """
    def __init__(self) -> None:
        DirectMethodWeight.__init__(self)
    
    def fit_tabular(self, data, pi_e, cfg):
        S = np.squeeze(data.states())
        SN = np.squeeze(data.next_states())
        PI0 = data.base_propensity()
        PI1 = data.target_propensity()
        REW = data.rewards()
        ACTS = data.actions()

        den_discrete = Density_Ratio_discounted(data.n_dim, cfg.gamma)

        for episode in range(len(S)):
            discounted_t = 1.0
            initial_state = S[episode][0]
            for (s,a,sn,r,pi1,pi0) in zip(S[episode],ACTS[episode],SN[episode], REW[episode], PI1[episode], PI0[episode]):
                discounted_t *= cfg.gamma
                policy_ratio = (pi1/pi0)[a]
                den_discrete.feed_data(s, sn, initial_state, policy_ratio, discounted_t)
            den_discrete.feed_data(-1, initial_state, initial_state, 1, 1-discounted_t)


        self.x, self.w = den_discrete.density_ratio_estimate()
        self.fitted = 'tabular'
          
    def evaluate_tabular(self, data, cfg):
        
        S = np.squeeze(data.states())
        SN = np.squeeze(data.next_states())
        PI0 = data.base_propensity()
        PI1 = data.target_propensity()
        REW = data.rewards()
        ACTS = data.actions()

        total_reward = 0.0
        self_normalizer = 0.0
        for episode in range(len(S)):
            discounted_t = 1.0
            for (s,a,sn,r,pi1,pi0) in zip(S[episode],ACTS[episode],SN[episode], REW[episode], PI1[episode], PI0[episode]):
                policy_ratio = (pi1/pi0)[a]
                total_reward += self.w[s] * policy_ratio * r * discounted_t
                self_normalizer += self.w[s] * policy_ratio * discounted_t
                discounted_t *= cfg.gamma

        return total_reward / self_normalizer

    def fit_NN(self, data, pi_e, config, verbose=True) -> float:
        """(Neural) Get the IH OPE estimate for pi_e.

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

        cfg = config.models['IH']
        processor = config.processor
        
        # TODO: early stopping + lr reduction
        
        im = data.states()[0]
        if processor: im = processor(im)
        self.W = cfg['model'](im.shape[1:], 1) # w(s) -> R
        optimizer = optim.Adam(self.W.parameters())

        print('Training: IH')
        losses = []
        
        batch_size = cfg['batch_size']    
        dataset_length = data.num_tuples()
        perm = np.random.permutation(range(dataset_length))
        eighty_percent_of_set = int(1.*len(perm))
        training_idxs = perm[:eighty_percent_of_set]
        validation_idxs = perm[eighty_percent_of_set:]
        training_steps_per_epoch = int(1. * np.ceil(len(training_idxs)/float(batch_size)))
        validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batch_size)))
        # steps_per_epoch = 1 #int(np.ceil(len(dataset)/float(batch_size)))
        for k in tqdm(range(cfg['max_epochs'])):
            train_gen = self.generator(data, config, training_idxs, fixed_permutation=True, batch_size=batch_size, processor=processor)
            val_gen = self.generator(data, config, training_idxs, fixed_permutation=True, batch_size=batch_size, processor=processor)

            M = 5
        
            for step in range(training_steps_per_epoch):
                
                with torch.no_grad():
                    inp, _ = next(train_gen)
                    state = torch.from_numpy(inp[0]).float()
                    next_state = torch.from_numpy(inp[1]).float()
                    policy_ratio = torch.from_numpy(inp[2]).float().view(-1, 1)
                    is_start = torch.from_numpy(inp[3]).float().view(-1, 1)
                    med_dist = torch.from_numpy(inp[4]).float()
                    
                    w_next = self.W.predict(next_state)
                    w_next = torch.exp(torch.clamp(w_next, -10, 10))


                w = self.W.predict(state)
                w = torch.exp(torch.clamp(w, -10, 10))

                
                # import pdb; pdb.set_trace()
                norm_w = w.mean()
                x = (1-is_start) * w * policy_ratio + is_start * norm_w - w_next
                x = x.view((-1, 1))
                diff_xx = torch.unsqueeze(next_state, 0) - torch.unsqueeze(next_state, 1)
                K_xx = torch.exp(-diff_xx.pow(2).sum(list(range(len(diff_xx.shape)))[2:]))/(2.0*med_dist*med_dist)
                loss_xx = torch.matmul(torch.matmul(torch.transpose(x, 0, 1), K_xx), x)
                loss = torch.squeeze(loss_xx) / (norm_w * norm_w)
                # loss = loss.mean()

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.W.parameters(), cfg['clipnorm'])
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
            obj1: [state, action, policy ratio, isStart, median distance for kernel]
            obj2: []
        """
        data_length = len(all_idxs)
        steps = int(np.ceil(data_length/float(batch_size)))

        n = len(data)
        T = max(data.lengths())
        n_dim = data.n_dim
        n_actions = data.n_actions


        S = np.hstack([data.states()[:,[0]], data.states()])
        SN = np.hstack([data.states()[:,[0]], data.next_states()])
        PI0 = np.hstack([data.base_propensity()[:,[0]], data.base_propensity()])
        PI1 = np.hstack([data.target_propensity()[:,[0]], data.target_propensity()])

        ACTS = np.hstack([np.zeros_like(data.actions()[:,[0]]), data.actions()])
        pi0 = []
        pi1 = []
        for i in range(len(ACTS)):
            pi0_ = []
            pi1_ = []
            for j in range(len(ACTS[1])):
                a = ACTS[i,j]
                pi0_.append(PI0[i,j,a])
                pi1_.append(PI1[i,j,a])
            pi0.append(pi0_)
            pi1.append(pi1_)

        PI0 = np.array(pi0)
        PI1 = np.array(pi1)


        REW = np.hstack([np.zeros_like(data.rewards()[:,[0]]), data.rewards()])
        ISSTART = np.zeros_like(REW)
        ISSTART[:,0] = 1.

        PROBS = np.repeat(np.atleast_2d(cfg.gamma**np.arange(-1,REW.shape[1]-1)), REW.shape[0], axis=0).reshape(REW.shape)

        S = np.vstack(S)
        SN = np.vstack(SN)
        PI1 = PI1.reshape(-1)
        PI0 = PI0.reshape(-1)
        ISSTART = ISSTART.reshape(-1)
        PROBS = PROBS.reshape(-1)
        PROBS /= sum(PROBS)

        N = S.shape[0]

        subsamples = np.random.choice(N, len(S))
        bs = batch_size
        num_batches = max(len(subsamples) // bs,1)
        med_dist = []
        for batch_num in tqdm(range(num_batches)):
            low_ = batch_num * bs
            high_ = (batch_num + 1) * bs
            sub = subsamples[low_:high_]
            # if self.modeltype in ['conv']:
            #     s = cfg.processor(S[sub])
            # else:
            #     s = S[sub].reshape(len(sub),-1)[...,None,None]
            s = cfg.processor(S[sub])

            med_dist.append(np.sum(np.square(s[None, :, :] - s[:, None, :]), axis = tuple([-3,-2,-1])))

        med_dist = np.sqrt(np.median(np.array(med_dist).reshape(-1)[np.array(med_dist).reshape(-1) > 0]))

        while True:
            # perm = np.random.permutation(all_idxs)
            for batch in np.arange(steps):
                # batch_idxs = perm[(batch*batch_size):((batch+1)*batch_size)]
                batch_idxs = np.random.choice(S.shape[0], batch_size, p=PROBS)

                # if self.modeltype in ['conv', 'conv1']:
                #     state = self.processor(S[batch_idxs])
                #     next_state = self.processor(SN[batch_idxs])
                # else:
                #     state = S[batch_idxs].reshape(len(batch_idxs),-1)#[...,None,None]
                #     next_state = SN[batch_idxs].reshape(len(batch_idxs),-1)#[...,None,None]
                state = cfg.processor(S[batch_idxs])
                next_state = cfg.processor(SN[batch_idxs])

                policy_ratio = PI1[batch_idxs] / PI0[batch_idxs]
                isStart = ISSTART[batch_idxs]
                median_dist = np.repeat(med_dist, batch_size)

                yield ([state,next_state,policy_ratio,isStart,median_dist], [])

    def evaluate_NN(self, data, cfg):
        
        S = data.states() #np.hstack([self.data.states()[:,[0]], self.data.states()])
        PI0 = data.base_propensity() #np.hstack([self.data.base_propensity()[:,[0]], self.data.base_propensity()])
        PI1 = data.target_propensity() #np.hstack([self.data.target_propensity()[:,[0]], self.data.target_propensity()])

        ACTS = data.actions() #np.hstack([np.zeros_like(self.data.actions()[:,[0]]), self.data.actions()])
        pi0 = []
        pi1 = []
        for i in range(len(ACTS)):
            pi0_ = []
            pi1_ = []
            for j in range(len(ACTS[1])):
                a = ACTS[i,j]
                pi0_.append(PI0[i,j,a])
                pi1_.append(PI1[i,j,a])
            pi0.append(pi0_)
            pi1.append(pi1_)

        PI0 = np.array(pi0)
        PI1 = np.array(pi1)

        REW = data.rewards() #np.hstack([np.zeros_like(self.data.rewards()[:,[0]]), self.data.rewards()])

        PROBS = np.repeat(np.atleast_2d(cfg.gamma**np.arange(REW.shape[1])), REW.shape[0], axis=0).reshape(REW.shape)

        S = np.vstack(S)
        PI1 = PI1.reshape(-1)
        PI0 = PI0.reshape(-1)
        PROBS = PROBS.reshape(-1)
        REW = REW.reshape(-1)

        predict_batch_size = 128
        steps = int(np.ceil(S.shape[0]/float(predict_batch_size)))
        densities = []
        for batch in np.arange(steps):
            batch_idxs = np.arange(S.shape[0])[(batch*predict_batch_size):((batch+1)*predict_batch_size)]

            s = cfg.processor(S[batch_idxs])
            densities.append(self.W.predict(torch.from_numpy(s).float()).detach().numpy())
            # if self.modeltype in ['conv', 'conv1']:
            #     s = cfg.processor(S[batch_idxs])
            #     densities.append(self.state_to_w.predict(s))
            # else:
            #     s = S[batch_idxs]
            #     s = s.reshape(s.shape[0], -1)
            #     densities.append(self.state_to_w.predict(s))

        densities = np.vstack(densities).reshape(-1)
        estimate = self.off_policy_estimator_density_ratio(REW, PROBS, PI1/PI0, densities)
        return estimate/np.sum(cfg.gamma ** np.arange(max(data.lengths())))

    @staticmethod
    def off_policy_estimator_density_ratio(rew, prob, ratio, den_r):
        return np.sum(prob * den_r * ratio * rew)/np.sum(prob * den_r * ratio)

def linear_solver(n, M):
    M -= np.amin(M) # Let zero sum game at least with nonnegative payoff
    c = np.ones((n))
    b = np.ones((n))
    res = linprog(-c, A_ub = M.T, b_ub = b)
    w = res.x
    return w/np.sum(w)

def quadratic_solver(n, M, regularizer):
    qp_G = np.matmul(M, M.T)
    qp_G += regularizer * np.eye(n)

    qp_a = np.zeros(n, dtype = np.float64)

    qp_C = np.zeros((n,n+1), dtype = np.float64)
    for i in range(n):
        qp_C[i,0] = 1.0
        qp_C[i,i+1] = 1.0
    qp_b = np.zeros(n+1, dtype = np.float64)
    qp_b[0] = 1.0
    meq = 1
    res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
    w = res[0]
    return w

class Density_Ratio_discounted(object):
    def __init__(self, num_state, gamma):
        self.num_state = num_state
        self.Ghat = np.zeros([num_state, num_state], dtype = np.float64)
        self.Nstate = np.zeros([num_state, 1], dtype = np.float64)
        self.initial_b = np.zeros([num_state], dtype = np.float64)
        self.gamma = gamma

    def reset(self):
        num_state = self.num_state
        self.Ghat = np.zeros([num_state, num_state], dtype = np.float64)
        self.Nstate = np.zeros([num_state, 1], dtype = np.float64)

    def feed_data(self, cur, next, initial, policy_ratio, discounted_t):
        if cur == -1:
            self.Ghat[next, next] -= discounted_t
        else:
            self.Ghat[cur, next] += policy_ratio * discounted_t
            self.Ghat[cur, initial] += (1-self.gamma)/self.gamma * discounted_t
            self.Ghat[next, next] -= discounted_t
            self.Nstate[cur] += discounted_t

    def density_ratio_estimate(self, regularizer = 0.001):
        Frequency = self.Nstate.reshape(-1)
        tvalid = np.where(Frequency >= 1e-20)
        G = np.zeros_like(self.Ghat)
        Frequency = Frequency/np.sum(Frequency)
        G[tvalid] = self.Ghat[tvalid]/(Frequency[:,None])[tvalid]
        n = self.num_state
        x = quadratic_solver(n, G/50.0, regularizer)
        w = np.zeros(self.num_state)
        w[tvalid] = x[tvalid]/Frequency[tvalid]
        return x, w



