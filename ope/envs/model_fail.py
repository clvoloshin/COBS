import numpy as np
import scipy.signal as signal

class MFail(object):
    def __init__(self,
                 make_pomdp = False,
                 transitions_deterministic=True,
                 max_length = 2,
                 sparse_rewards = False,
                 stochastic_rewards = False):

        self.allowable_actions = [0,1]
        self.n_actions = len(self.allowable_actions)
        self.n_dim = 3



        self.make_pomdp = make_pomdp

        self.state_to_pomdp_state = {}
        self.state_to_pomdp_state[0] = 0
        self.state_to_pomdp_state[1] = 1
        self.state_to_pomdp_state[2] = 1
        self.state_to_pomdp_state[3] = 2

        # self.state_to_pomdp_state[0] = 0
        # self.state_to_pomdp_state[2*max_length-1] = number_of_pomdp_states-1

        print(self.state_to_pomdp_state)
        self.transitions_deterministic = transitions_deterministic
        self.slippage = .25
        self.max_length = max_length
        self.sparse_rewards = sparse_rewards
        self.stochastic_rewards = stochastic_rewards
        self.reward_overwrite = None # only for simulator work
        self.absorbing_state = None # only for simulator work
        self.reset()

    def overwrite_rewards(self, new_r):
        self.reward_overwrite = new_r

    def set_absorb(self, absorb):
        self.absorbing_state = absorb

    def num_states(self):
        return self.n_dim

    def pos_to_image(self, x):
        '''latent state -> representation '''
        return x

    def reset(self):
        self.state = 0
        self.done = False
        self.T = 0
        return np.array([self.state])

    def step(self, action):

        p = .25

        assert action in self.allowable_actions
        assert not self.done, 'Episode Over'
        reward = 0 if not self.stochastic_rewards else np.random.randn()
        prev_state = self.state_to_pomdp_state[self.state] if self.make_pomdp else self.state
        self.T += 1

        if self.state == 0:
            reward = 0
            if action == 0:
                self.state = np.random.choice([1, 2], p=[p, 1-p])
            else:
                self.state = np.random.choice([1, 2], p=[1-p, p])
        else:
            reward = 1 if self.state == 1 else -1
            self.state = 0
            if self.T >= self.max_length:
                self.done = True

        state = self.state_to_pomdp_state[self.state] if self.make_pomdp else self.state

        if self.reward_overwrite is not None:
            key = tuple([int(prev_state), int(action), int(state)]) if not self.done else tuple([prev_state, action, self.absorbing_state])
            # key = tuple([int(prev_state), int(action)]) if not self.done else tuple([prev_state, action])
            if key in self.reward_overwrite:
                try:
                    reward = np.random.choice(list(self.reward_overwrite[key]), p=list(self.reward_overwrite[key].values()))
                except:
                    import pdb; pdb.set_trace()
            else:
                print(key)
                reward = 0

        if self.make_pomdp:
            # only reveal state, not internal state (POMDP)
            return np.array([state]), reward, self.done, {}
        else:
            return np.array([self.state]), reward, self.done, {}

    def render(self, a=None, r=None, return_arr=False):
        start_state = 1 if self.state == 0 else 0
        state = np.zeros(2*self.max_length-2)
        end_state = 1 if self.state == (2*self.max_length-1) else 0

        if not start_state and not end_state:
            state[self.state-1] = 1

        if return_arr:
            return start_state, state.reshape(2,self.max_length-1, order='F'), end_state
        else:

            print(' ', ' '.join(state.reshape(2,self.max_length-1, order='F')[0].astype(int).astype(str).tolist()), '  ')
            if (a is not None) and (r is not None):
                print(start_state, ' '*((2*(self.max_length-2))+1), end_state, ' (a,r): ', (a,r), '.  If POMDP, End state: ', end_state)
            else:
                print(start_state, ' '*((2*(self.max_length-2))+1), end_state)
            print(' ', ' '.join(state.reshape(2,self.max_length-1, order='F')[1].astype(int).astype(str).tolist()), '  ')
            print('\n')
            # print([start_state], [end_state], state.reshape(2,self.max_length-1, order='F'), )

    def calculate_exact_value_of_policy(self, pi_e, gamma):
        # Exact
        # rewards = []
        # if (self.transitions_deterministic):
        #   rew = [(+1)*(pi_e.probs[0]) + (-1)*(pi_e.probs[1])]
        #   if not self.sparse_rewards:
        #       rewards.append(rew*self.max_length)
        #   else:
        #       rewards.append([0]*(self.max_length-1) + rew)

        # else:
        #   rew = [(+1)*(pi_e.probs[0]*(1-self.slippage) + pi_e.probs[1]*(self.slippage)) + (-1)*(pi_e.probs[0]*(self.slippage) + pi_e.probs[1]*(1-self.slippage))]
        #   if not self.sparse_rewards:
        #       rewards.append(rew*self.max_length)
        #   else:
        #       rewards.append([0]*(self.max_length-1) + rew)

        # Approx
        evaluation = []
        for i in range(5000):
            done = False
            state = self.reset()
            # env.render()
            rewards = []
            while not done:
                action = pi_e([state])
                # print(action)
                next_state, reward, done = self.step(action)
                # env.render()
                state = next_state
                rewards.append(reward)

            evaluation.append(rewards)

        true = np.mean([self.discounted_sum(rew, gamma) for rew in np.array(evaluation)])

        return true

    @staticmethod
    def discounted_sum(costs, discount):
        '''
        Calculate discounted sum of costs
        '''
        y = signal.lfilter([1], [1, -discount], x=costs[::-1])
        return y[::-1][0]
