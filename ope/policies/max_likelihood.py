import numpy as np
import scipy.signal as signal

class MaxLikelihoodModel(object):
    def __init__(self, gamma, max_traj_length=None, action_space_dim = 2):
        self.gamma = gamma
        self.override_done = True if max_traj_length is not None else False
        self.max_traj_length = max_traj_length
        self.action_space_dim = action_space_dim

    def run(self, dataset):
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

    def transition(self, x, a):
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

    def Q(self, policy, x, t=0):
        all_Qs = []
        for t, X in enumerate(x):
            Qs = []
            for a in range(self.action_space_dim):
                state = X[0][0]
                if isinstance(a, type(np.array([]))) or isinstance(a, list):
                    assert len(a) == 1
                    a = a[0]
                done = False
                costs = []
                trajectory_length = t
                # Q
                while not done:

                    new_state, done = self.transition(state, a)
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

                Qs.append(self.discounted_sum(costs, self.gamma))
            all_Qs.append(Qs)
        return np.array(all_Qs)

    def V(self, policy, x, t=0):
        state = x
        done = False
        weighted_costs = []
        trajectory_length = t
        # V
        while not done:

            a = policy([state])[0]
            r = sum([prob*self.estimate_R(state, act, trajectory_length) for act, prob in enumerate(policy.predict([state])[0]) ])

            new_state, done = self.transition(state, a)
            weighted_costs.append( r )

            # if (tuple([state,a,new_state]) in self.terminal_transitions):
            #     done = True

            # print(state,a,r,new_state,done)
            trajectory_length += 1
            if self.max_traj_length is not None:
                if trajectory_length >= self.max_traj_length:
                    done = True

            state = new_state

        return self.discounted_sum(weighted_costs, self.gamma)

    @staticmethod
    def discounted_sum(costs, discount):
        '''
        Calculate discounted sum of costs
        '''
        y = signal.lfilter([1], [1, -discount], x=costs[::-1])
        return y[::-1][0]
