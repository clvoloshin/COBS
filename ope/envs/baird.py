import numpy as np
import scipy.signal as signal
from numpy.random import RandomState

class Baird(object):
    def __init__(self, *args, **kw):
        
        self.rng = self.set_seed(0)
        self.allowable_actions = [0, 1]
        self.n_actions = len(self.allowable_actions)
        self.n_dim = 7+1 #+ terminal
        self.terminal = 7
        
    @staticmethod
    def set_seed(seed=None):
        if seed is not None:
            rng = RandomState(seed)
        else:
            rng = RandomState()
        return rng

    def num_states(self):
        return self.n_dim

    def pos_to_image(self, x):
        '''latent state -> representation '''
        return x

    def reset(self):
        self.state = self.rng.randint(7)
        self.done = False
        return np.array([self.state])

    def step(self, action):
        assert action in self.allowable_actions
        assert not self.done, 'Episode Over'
        
        if self.state == 6:
            self.done = True if self.rng.rand() < .05 else False
        if self.done:
            self.state = self.terminal # terminal state
        else:
            if action == 0:
                self.state = self.rng.randint(6)
            else:
                self.state = 6

        return np.array([self.state]), 0, self.done, {}

    def processor(self, state):
        X = np.array([[2, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 2, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 2, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 2, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 2, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 2, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 2, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return np.atleast_2d(X[state])

    @staticmethod
    def discounted_sum(costs, discount):
        '''
        Calculate discounted sum of costs
        '''
        y = signal.lfilter([1], [1, -discount], x=costs[::-1])
        return y[::-1][0]

