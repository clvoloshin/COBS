import numpy as np
from collections import OrderedDict

class Gridworld(object):
    def __init__(self, slippage=0.0, start_on_border=True):
        h = -0.5
        f = -0.005
        self.grid = np.array(
            [[-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
             [-0.01, -0.01, f, -0.01, h, -0.01, -0.01, -0.01],
             [-0.01, -0.01, -0.01, h, -0.01, -0.01, f, -0.01],
             [-0.01, f, -0.01, -0.01, -0.01, h, -0.01, f],
             [-0.01, -0.01, -0.01, h, -0.01, -0.01, f, -0.01],
             [-0.01, h, h, -0.01, f, -0.01, h, -0.01],
             [-0.01, h, -0.01, -0.01, h, -0.01, h, -0.01],
             [-0.01, -0.01, -0.01, h, -0.01, f, -0.01, +1]])

        self.start_on_border = start_on_border
        self.dynamics_propensity = None
        self.sup_dynamics_propensity = None
        self.dynamics_propensity_as_dict = None
        self.terminal_state = 63
        self.n_actions = 4
        self.slippage = slippage
        self.n_dim = np.prod(self.grid.shape) + 1  # +1 for absorbing state
        # self.mapping_reversed = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3} #old system
        self.mapping_reversed = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
        self.mapping = {val: key for key, val in self.mapping_reversed.items()}
        self.reset()

    def set_new_grid(self, grid, slippage=0.0):
        self.grid = grid

        self.terminal_state = self.grid.size - 1
        self.n_actions = 4
        self.slippage = slippage
        self.n_dim = np.prod(self.grid.shape) + 1  # +1 for absorbing state
        # self.mapping_reversed = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3} #old system
        self.mapping_reversed = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
        self.mapping = {val: key for key, val in self.mapping_reversed.items()}
        self.reset()

    def set_reward_function(self, new_R):
        if (new_R.shape[0] != self.grid.shape[0]) or  (new_R.shape[1] != self.grid.shape[1]):
            print("Reward function unchanged because shape is not correct. Expected: %s. Received %s." % (self.grid.shape, new_R.shape))
        else:
            self.grid = new_R

    def set_dynamics_propensity(self, dynamics_propensity, sup, dic):
        self.dynamics_propensity = dynamics_propensity
        self.sup_dynamics_propensity = sup
        self.dynamics_propensity_as_dict = dic

    def expected_utility(self, a, s, U):
        return sum([p * U[s1] for (p, s1) in self.T(s, a)])

    def best_policy(self, epsilon=.001):
        U = self.value_iteration(epsilon)
        pi = {}
        for s in range(self.n_dim - 1):
            pi[s] = np.argmax([
                self.expected_utility(a, s, U) for a in range(self.n_actions)
            ])
            # pi[s] = np.argmax(range(self.n_actions), lambda a: self.expected_utility(a, s, U))
        return pi

    def value_iteration(self, epsilon, gamma=.99):
        U1 = dict([(s, 0) for s in range(self.n_dim - 1)])
        while True:
            U = U1.copy()
            delta = 0
            for s in range(self.n_dim - 1):

                state = (s // self.grid.shape[0], s % self.grid.shape[1])

                # [sum([p * U[s1] for (p, s1) in T(s, a)])
                #                            for a in self.mapping.keys()]

                r = self.grid[state[0], state[1]]

                probabilities_by_state = [
                    self.T(s, a) for a in np.arange(self.n_actions)
                ]

                U1[s] = r + gamma * max([
                    sum([p * U[s1] for (p, s1) in lst])
                    for lst in probabilities_by_state
                ]) * (s != self.terminal_state)
                delta = max(delta, abs(U1[s] - U[s]))

            if delta < epsilon * (1 - gamma) / gamma:
                return U

    def Q_pi(self, policy, gamma, epsilon = .0001):
        # Evaluate Q^\pi as if <X,A,R,P> were known.
        def get_R(s,a,s_):
            try:
                return self.grid[s_ // self.grid.shape[1], s_ % self.grid.shape[0]]
            except:
                return 0

        U1 = np.zeros((self.n_dim-1, self.n_actions))
        while True:
            U = U1.copy()
            for state in range(self.n_dim-1):
                for action in range(self.n_actions):

                    state_is_terminal = state != self.terminal_state

                    if self.dynamics_propensity is None:
                        T = self.T(state, action, use_slippage = True)
                    else:
                        T = OrderedDict()
                        for val, key in self.T(state, action, use_slippage=True):
                            if key in T:
                                T[key] += val
                            else:
                                T[key] = val

                        probs = {key: val * (self.dynamics_propensity(state, action, key))/self.sup_dynamics_propensity for key, val in T.items()}
                        if any(probs.values()):
                            norm = sum(probs.values())
                            probs = {key:prob/norm for key,prob in probs.items()}
                            T = [[prob, key] for key, prob in probs.items()]
                        else:
                            T = [[prob, key] for key, prob in T.items()]
                            # U1[state, action] = 0
                            # continue

                    Rs = [p_*state_is_terminal* get_R(state,action,s_) for p_,s_ in T]
                    R = sum(Rs)

                    expected_Q_s_a_ = sum([p_s_* state_is_terminal *sum([p_a_ * U[s_,a_] for a_,p_a_ in enumerate(policy.predict([s_])[0])]) for p_s_,s_ in T])
                    U1[state, action] = R + gamma*expected_Q_s_a_

            delta = np.linalg.norm(U1 - U)
            if delta < epsilon:
                return U

    def T(self, s, a, use_slippage=False):
        '''
        Used to calc best policy. However, for the purposes of the paper,
        we calc a policy assuming no slippage even though the actual dynamics
        have slippage. For the actual T, set the use_slippage flag to True.
        '''
        state = (s // self.grid.shape[0], s % self.grid.shape[1])

        out = []
        for action in np.arange(self.n_actions):
            new_state = self.vector_add(state, self.mapping[action])
            if (not self.is_valid(new_state)) or (s >= self.terminal_state):
                new_state = state

            slippage = self.slippage if use_slippage else 0
            if action == a:
                out.append([
                    1 - slippage + slippage / self.n_actions,
                    new_state[0] * self.grid.shape[0] + new_state[1]
                ])
            else:
                out.append([
                    slippage / self.n_actions,
                    new_state[0] * self.grid.shape[0] + new_state[1]
                ])

        return out
        # new_state = self.vector_add(state, a)
        # return 1-self.slippage + self.slippage/self.n_actions, new_state[0]*self.grid.shape[0] + new_state[1]

    def num_states(self):
        return self.n_dim

    @staticmethod
    def vector_add(x, y):
        return (x[0] + y[0], x[1] + y[1])

    def is_valid(self, x):
        return (0 <= x[0] < self.grid.shape[0]) and (0 <= x[1] <
                                                     self.grid.shape[1])

    def step(self, action):
        assert not self.done

        if self.dynamics_propensity is None:

            possible_action = np.random.choice(self.n_actions)
            action = np.random.choice([action, possible_action],
                                      p=[1 - self.slippage, self.slippage])
            action_tuple = self.mapping[action]

            state_tuple = (self.state // self.grid.shape[1], self.state % self.grid.shape[1])
            new_state_tuple = self.vector_add(state_tuple, action_tuple)

        else:
            T = OrderedDict()
            state = self.state[0]
            for val, key in self.T(state, action, use_slippage=True):
                if key in T:
                    T[key] += val
                else:
                    T[key] = val

            probs = [val * (self.dynamics_propensity(state, action, key))/self.sup_dynamics_propensity for key, val in T.items()]
            if any(probs):
                norm = sum(probs)
                probs = [prob/norm for prob in probs]
                next_state = np.random.choice([key for key in T.keys()], p=probs)
            else:
                next_state = np.random.choice([key for key in T.keys()], p=[val for val in T.values()])
                return self.state, reward, self.done, {}

            new_state_tuple = (np.array([next_state // self.grid.shape[1]]), np.array([next_state % self.grid.shape[0]]))

        if not self.is_valid(new_state_tuple):
            new_state_tuple = state_tuple

        self.state = new_state_tuple[0] * self.grid.shape[0] + new_state_tuple[
            1]

        if self.state == self.terminal_state:
            self.done = True

        reward = self.grid[new_state_tuple[0], new_state_tuple[1]][0]

        return self.state, reward, self.done, {}

    def reset(self):
        # self.state = np.array([0])
        # self.init_pos = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56])
        if self.start_on_border:
            self.init_pos = np.union1d(np.arange(self.grid.shape[1]), self.grid.shape[1] * np.arange(self.grid.shape[0]))
        else:
            self.init_pos = np.array([0])
        self.state = np.random.choice(self.init_pos, 1)
        self.done = False
        return self.state

    def render_policy(self, policy):
        chars = {
            (0, 1): '>',
            (-1, 0): '^',
            (0, -1): '<',
            (1, 0): 'v',
            None: '.'
        }
        print(
            np.array([[
                chars[self.mapping[policy[row * self.grid.shape[1] + col]]]
                for col in range(self.grid.shape[0])
            ] for row in range(self.grid.shape[1])]))

    def render(self):
        print("Not Implemented")

    def get_num_states(self):
        raise NotImplemented
