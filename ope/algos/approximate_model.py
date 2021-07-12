
from ope.algos.direct_method import DirectMethodModelBased
import numpy as np
import scipy.signal as signal
from ope.utls.thread_safe import threadsafe_generator
import os
import time
from copy import deepcopy
from sklearn.linear_model import LinearRegression, LogisticRegression
from tqdm import tqdm

import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

class ApproxModel(DirectMethodModelBased):
    """Algorithm: Approx Model (Model-Based). 

    This is class builds a model from which Q can be estimated through rollouts.
    """
    def __init__(self, cfg, n_actions):
        DirectMethodModelBased.__init__(self)
        self.frameheight = cfg.frameheight
        self.frameskip = cfg.frameskip
        self.max_traj_length = cfg.models['MBased']['max_traj_length']
        self.override_done = True if self.max_traj_length is not None else False
        self.n_actions = n_actions
 
    def fit_NN(self, data, pi_e, config, verbose=True) -> float:
        cfg = config.models['MBased']
        processor = config.processor
        
        # TODO: early stopping + lr reduction
        
        im = data.states()[0]
        if processor: im = processor(im)
        self.model = cfg['model'](im.shape[1:], data.n_actions)
        optimizer = optim.Adam(self.model.parameters())
        
        print('Training: Model Free')
        losses = []
        
        batch_size = cfg['batch_size']

        dataset_length = data.num_tuples()
        perm = np.random.permutation(range(dataset_length))
        eighty_percent_of_set = int(.8*len(perm))
        training_idxs = perm[:eighty_percent_of_set]
        validation_idxs = perm[eighty_percent_of_set:]
        training_steps_per_epoch = int(1.*np.ceil(len(training_idxs)/float(batch_size)))
        validation_steps_per_epoch = int(np.ceil(len(validation_idxs)/float(batch_size)))
        
        for k in tqdm(range(cfg['max_epochs'])):
            train_gen = self.generator(data, config, training_idxs, fixed_permutation=True, batch_size=batch_size, processor=processor)
            val_gen = self.generator(data, config, training_idxs, fixed_permutation=True, batch_size=batch_size, processor=processor)

            M = 5
        
            for step in range(training_steps_per_epoch):
                
                with torch.no_grad():
                    inp, out = next(train_gen)
                    states = torch.from_numpy(inp[0]).float()
                    actions = torch.from_numpy(inp[1]).bool()
                    
                    next_states = torch.from_numpy(out[0]).float()
                    rewards = torch.from_numpy(out[1]).float()
                    dones = torch.from_numpy(out[2]).float()

                pred_next_states, pred_rewards, pred_dones = self.model(states, actions)

                states_loss = (pred_next_states - next_states).pow(2).mean()
                rewards_loss = (pred_rewards - rewards).pow(2).mean()
                dones_loss = nn.BCELoss()(pred_dones, dones)
                loss = states_loss + rewards_loss + dones_loss
                
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg['clipnorm'])
                optimizer.step()

        self.fitted = 'NN'
        return 1.0
    
    @threadsafe_generator
    def generator(self, data, cfg, all_idxs, fixed_permutation=False,  batch_size = 64, processor=None):
        """Data Generator for Model-Based 

        Parameters
        ----------
        env : obj
            The environment object.
        all_idxs : ndarray
            1D array of ints representing valid datapoints from which we generate examples
        batch_size : int
            Minibatch size to during training

        
        Yield
        -------
        obj1, obj2
            obj1: [state, action]
            obj2: [next state, reward, is done]
        """


        data_length = len(all_idxs)
        steps = int(np.ceil(data_length/float(batch_size)))

        states = data.states()
        states_ = data.next_states()
        lengths = data.lengths()
        rewards = data.rewards().reshape(-1)
        actions = data.actions().reshape(-1)
        dones = data.dones().reshape(-1)

        

        shp = states.shape
        states = states.reshape(np.prod(shp[:2]), -1)
        states_ = states_.reshape(np.prod(shp[:2]), -1)

        while True:
            perm = np.random.permutation(all_idxs)
            for batch in np.arange(steps):
                batch_idxs = perm[(batch*batch_size):((batch+1)*batch_size)]

                x = states[batch_idxs]
                x_ = states_[batch_idxs]
                r = rewards[batch_idxs]
                done = dones[batch_idxs]
                act = actions[batch_idxs]

                tmp_shp = np.hstack([len(batch_idxs),-1,shp[2:]])
                inp = cfg.processor(x.reshape(tmp_shp).squeeze())
                inp = inp[:,None,:,:]
                out_x_ = np.squeeze((x_-x).reshape(tmp_shp))
                out_x_ = out_x_[:,None,:,:]
                out_r = -r
                out_done = done

                # if self.modeltype in ['conv']:
                #     tmp_shp = np.hstack([len(batch_idxs),-1,shp[2:]])
                #     inp = self.processor(x.reshape(tmp_shp).squeeze())
                #     out_x_ = np.diff(self.processor(x_.reshape(tmp_shp)).squeeze(), axis=1)[:,[-1],...]
                #     out_r = -r
                #     out_done = done
                # elif self.modeltype == 'conv1':
                #     tmp_shp = np.hstack([len(batch_idxs),-1,shp[2:]])
                #     inp = self.processor(x.reshape(tmp_shp).squeeze())
                #     inp = inp[:,None,:,:]
                #     out_x_ = np.squeeze((x_-x).reshape(tmp_shp))
                #     out_x_ = out_x_[:,None,:,:]
                #     out_r = -r
                #     out_done = done
                # else:
                #     tmp_shp = np.hstack([len(batch_idxs),-1,shp[2:]])
                #     inp = np.squeeze(x.reshape(tmp_shp))
                #     out_x_ = x_
                #     out_x_ = np.diff(out_x_.reshape(tmp_shp), axis=2).reshape(-np.prod(tmp_shp[:2]), -1)
                #     out_r = -r
                #     out_done = done
                #     out_x_ = out_x_[:,None,...]

                yield [inp, np.eye(data.n_actions)[act]], [out_x_, out_r, out_done]
                # yield ([x, np.eye(3)[acts], np.array(weight).reshape(-1,1)], [np.array(R).reshape(-1,1)])

    def transition_NN(self, x, a):
        # if isinstance(self.full, list):
        #     state_diff, r, prob_done = [model.predict(np.hstack([x.reshape(x.shape[0],-1), a])) for model in self.full]
        #     state_diff = state_diff[:,None,:]
        #     prob_done = [[d] for d in prob_done]
        # else:
        #     
        state_diff, r, prob_done = self.model(torch.from_numpy(x).float(), torch.from_numpy(a).bool())
        state_diff = state_diff.detach().numpy()
        r = r.detach().numpy()
        prob_done = prob_done.detach().numpy()

        x_ = np.concatenate([x[:,1:self.frameheight,...], x[:,(self.frameheight-1):self.frameheight,...] + state_diff], axis=1)
        done = np.array([np.random.choice([0,1], p=[1-d, d]) for d in prob_done])

        return x_, -r.reshape(-1), done

    def Q_NN(self, policy, x, gamma, t=0):
        """(Linear/Neural) Return the Model-Based OPE estimate for pi_e starting from a state

        Parameters
        ----------
        policy : obj
            A policy object, evaluation policy.
        x : ndarray
            State.
        t : int, optional
            time
            Default: 0

        Returns
        -------
        list
            The Q value starting from state x and taking each possible action
            in the action space:
                [Q(x, a) for a in A]
        """

        Qs = []

        # state = x
        # make action agnostic.

        state = np.repeat(x, self.n_actions, axis=0)
        acts  = np.tile(np.arange(self.n_actions), len(x))

        done = np.zeros(len(state))
        costs = []
        trajectory_length = t
        # Q
        cost_to_go = np.zeros(len(state))


        new_state, cost_holder, new_done = self.transition_NN(state, np.atleast_2d(np.eye(self.n_actions)[acts]))
        # cost_holder = self.estimate_R(state, np.atleast_2d(np.eye(self.action_space_dim)[acts]), None)

        done = done + new_done
        new_cost_to_go = cost_to_go + gamma * cost_holder * (1-done)


        norm_change = np.sqrt(np.sum((new_cost_to_go-cost_to_go)**2) / len(state))
        # print(trajectory_length, norm_change, cost_to_go, sum(done), len(done))
        cost_to_go = new_cost_to_go

        if norm_change < 1e-4:
            done = np.array([True])

        trajectory_length += 1
        if self.max_traj_length is not None:
            if trajectory_length >= self.max_traj_length:
                done = np.array([True])

        state = new_state

        while not done.all():
            tic=time.time()

            still_alive = np.where(1-done)[0]
            acts = policy.sample(state[still_alive])

            new_state, cost_holder, new_done = self.transition_NN(state[still_alive], np.atleast_2d(np.eye(self.n_actions)[acts]))

            # cost_holder = self.estimate_R(state, np.atleast_2d(np.eye(self.action_space_dim)[acts]), trajectory_length)
            # if (tuple([state,a,new_state]) in self.terminal_transitions):
            #     done = True

            done[still_alive] = (done[still_alive] + new_done).astype(bool)

            new_cost_to_go = cost_to_go[still_alive] + gamma * cost_holder * (1-done[still_alive])


            # norm_change = np.sqrt(np.sum((new_cost_to_go-cost_to_go)**2) / len(state))
            # print(trajectory_length, norm_change, cost_to_go, sum(done), len(done))
            cost_to_go[still_alive] = new_cost_to_go

            # if norm_change < 1e-4:
            #     done = np.array([True])

            trajectory_length += 1
            if self.max_traj_length is not None:
                if trajectory_length >= self.max_traj_length:
                    done = np.array([True])

            # print(time.time()-tic, trajectory_length)
            state[still_alive] = new_state

        return cost_to_go


    def fit_tabular(self, dataset, pi_e, config, verbose=True) -> float:
        '''
        probability of
        transitioning from s to s'
        given action a is the number of
        times this transition was observed divided by the number
        of times action a was taken in state s. If D contains no examples
        of action a being taken in state s, then we assume
        that taking action a in state s always causes a transition to
        the terminal absorbing state.
        '''
        # frames = np.array([x['frames'] for x in dataset])
        # transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
        #                           np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
        #                           frames[:,1:].reshape(-1), #np.array([x['x_prime'] for x in dataset]).reshape(-1,1).T
        #                           # np.array([x['done'] for x in dataset]).reshape(-1,1).T
        #                          ]).T

        frames = dataset.frames()
        transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                  dataset.actions(False) ,
                                  frames[:,1:].reshape(-1), #np.array([x['x_prime'] for x in dataset]).reshape(-1,1).T
                                  # np.array([x['done'] for x in dataset]).reshape(-1,1).T
                                 ]).T

        unique, idx, count = np.unique(transitions, return_index=True, return_counts=True, axis=0)

        # partial_transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
        #                           np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
        #                          ]).T

        partial_transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                  dataset.actions(False) ,
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

        # if self.absorbing is not None:
        #     for act in np.arange(self.action_space_dim):
        #         prob[tuple([self.absorbing[0], act])] = {self.absorbing[0]:1.}

        self.P = prob

        # all_transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
        #                               np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
        #                               frames[:,1:].reshape(-1), #np.array([x['x_prime'] for x in dataset]).reshape(-1,1).T ,
        #                               np.array([x['done'] for x in dataset]).reshape(-1,1).T ,
        #                          ]).T
        all_transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                      dataset.actions(False) ,
                                      frames[:,1:].reshape(-1), #np.array([x['x_prime'] for x in dataset]).reshape(-1,1).T ,
                                      dataset.dones(False) ,
                                 ]).T

        unique, idx, count = np.unique(all_transitions, return_index=True, return_counts=True, axis=0)


        unique_a_given_x, idx_a_given_x, count_a_given_x = np.unique(transitions, return_index=True, return_counts=True, axis=0)

        # key=(state, action). value= number of times a was taking in state
        all_counts_a_given_x = {tuple(key):value for key,value in zip(unique_a_given_x,count_a_given_x)}

        done = {}
        for idx,row in enumerate(unique):
            if tuple(row[:-1]) in done:
                done[tuple(row[:-1])][row[-1]] = count[idx] / all_counts_a_given_x[tuple(row[:-1])]
            else:
                done[tuple(row[:-1])] = {}
                done[tuple(row[:-1])][row[-1]] = count[idx] / all_counts_a_given_x[tuple(row[:-1])]
        self.D = done
        # self.terminal_transitions = {tuple([x,a,x_prime]):1 for x,a,x_prime in all_transitions[all_transitions[:,-1] == True][:,:-1]}

        # Actually fitting R, not Q_k
        # self.Q_k = self.model #init_Q(model_type=self.model_type)
        # X_a = np.array(zip(dataset['x'],dataset['a']))#dataset['state_action']
        # x_prime = dataset['x_prime']
        # index_of_skim = self.skim(X_a, x_prime)
        # self.fit(X_a[index_of_skim], dataset['cost'][index_of_skim], batch_size=len(index_of_skim), verbose=0, epochs=1000)
        # self.reward = self
        # self.P = prob

        if self.override_done:
            # transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
            #                           np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
            #                           np.array([range(len(x['x'])) for x in dataset]).reshape(-1,1).T,
            #                           np.array([x['r'] for x in dataset]).reshape(-1,1).T ,
            #                          ]).T

            # partial_transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
            #                       np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
            #                       np.array([range(len(x['x'])) for x in dataset]).reshape(-1,1).T,
            #                      ]).T
            transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                      dataset.actions(False) ,
                                      dataset.ts(False),
                                      dataset.rewards(False) ,
                                     ]).T

            partial_transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                  dataset.actions(False) ,
                                  dataset.ts(False),
                                 ]).T
        else:
            # transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
            #                           np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
            #                           np.array([x['r'] for x in dataset]).reshape(-1,1).T ,
            #                          ]).T

            # partial_transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
            #                       np.array([x['a'] for x in dataset]).reshape(-1,1).T ,
            #                      ]).T
            transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                      dataset.actions(False) ,
                                      dataset.rewards(False) ,
                                     ]).T

            partial_transitions = np.vstack([ frames[:,:-1].reshape(-1), #np.array([x['x'] for x in dataset]).reshape(-1,1).T ,
                                  dataset.actions(False) ,
                                 ]).T

        unique, idxs, counts = np.unique(transitions, return_index=True, return_counts=True, axis=0)

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
        self.fitted = 'tabular'

    def estimate_R(self, x, a, t):
        # Exact R
        # self.R = {(0, 0): {-1: .06, 0: .5, 1: .44}, (0, 1): {-1: .06, 0: .5, 1: .44}}

        #Approximated rewards
        if len(list(self.R)[0]) == 3:
            key = tuple([x,a,t])
        else:
            key = tuple([x,a])

        if key in self.R:
            try:
                reward = np.random.choice(list(self.R[key]), p=list(self.R[key].values()))
            except:
                import pdb; pdb.set_trace()
        else:
            reward = 0

        return reward

    def transition_tabular(self, x, a):
        # Exact MDP dynamics
        # self.P = {(0, 0): {0: 0.5, 1: 0.5}, (0, 1): {0: 0.5, 1: .5}}

        #Approximated dynamics
        if tuple([x,a]) in self.P:
            try:
                state = np.random.choice(list(self.P[(x,a)]), p=list(self.P[(x,a)].values()))
                if self.override_done:
                    done = False
                else:
                    done = np.random.choice(list(self.D[(x,a,state)]),
                                            p=list(self.D[(x,a,state)].values()))
            except:
                import pdb; pdb.set_trace()
        else:
            state = None
            done = True

        return state, done

    def Q_tabular(self, policy, x, gamma, t=0):
        all_Qs = []
        for t, X in enumerate(x):
            Qs = []
            for a in range(self.n_actions):
                state = X[0][0]
                if isinstance(a, type(np.array([]))) or isinstance(a, list):
                    assert len(a) == 1
                    a = a[0]
                done = False
                costs = []
                trajectory_length = t
                # Q
                while not done:

                    new_state, done = self.transition_tabular(state, a)
                    costs.append( self.estimate_R(state, a, trajectory_length) )
                    # if (tuple([state,a,new_state]) in self.terminal_transitions):
                    #     done = True

                    trajectory_length += 1
                    if self.max_traj_length is not None:
                        if trajectory_length >= self.max_traj_length:
                            done = True

                    if not done:
                        state = new_state
                        a = policy([state])[0]

                Qs.append(self.discounted_sum(costs, gamma))
            all_Qs.append(Qs)
        return np.array(all_Qs)    


    @staticmethod
    def discounted_sum(costs, discount):
        '''
        Calculate discounted sum of costs
        '''
        y = signal.lfilter([1], [1, -discount], x=costs[::-1])

        return y[::-1][0]

