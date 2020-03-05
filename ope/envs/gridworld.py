import numpy as np


class Gridworld(object):
    def __init__(self, slippage=0.0):
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

        self.terminal_state = 63
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

    def value_iteration(self, epsilon):
        gamma = .99
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
            if not self.is_valid(new_state):
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

        possible_action = np.random.choice(self.n_actions)
        action = np.random.choice([action, possible_action],
                                  p=[1 - self.slippage, self.slippage])
        action_tuple = self.mapping[action]

        state_tuple = (self.state // 8, self.state % 8)
        new_state_tuple = self.vector_add(state_tuple, action_tuple)

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
        init_pos = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56])
        self.state = np.random.choice(init_pos, 1)
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
