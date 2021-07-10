import numpy as np
import scipy.signal as signal

class EGreedyPolicy(object):
    def __init__(self, model, prob_deviation=0., action_space_dim = 3, processor=None, action_map = None):
        self.model = model
        self.prob_deviation = prob_deviation
        self.action_space_dim = action_space_dim
        self.processor = processor
        self.action_map = action_map

    def sample(self, X):

        if self.processor is not None:
            X = np.array([self.processor(x) for x in X])
        eps = np.random.random(size=len(X))
        mask = eps <= self.prob_deviation
        random_actions = np.random.choice(self.action_space_dim, size=len(X))
        model_acts = np.argmax(self.model.predict(X, batch_size=128), axis=1)
        model_acts[mask] = random_actions[mask]

        if len(model_acts) == 1:
            return model_acts[0]
        else:
            return np.array(model_acts)

        # act = []
        # for x in X:
        #     eps = np.random.random()
        #     if eps <= self.prob_deviation:
        #         act += [np.random.choice(self.action_space_dim)]
        #     else:
        #         act += [np.argmax(self.model.predict(x[np.newaxis, ...]), axis=1)[0]]

        # if len(act) == 1:
        #     return act[0]
        # else:
        #     return np.array(act)
    def get_action(self, action):
        if self.action_map is not None:
            if action in self.action_map:
                return self.action_map[action]
            else:
                return self.action_map['default']
        else:
            return action



    def predict(self, X):

        probs = np.ones((len(X), self.action_space_dim)) * (self.prob_deviation/self.action_space_dim)
        for idx, x in enumerate(np.atleast_1d(X)):
            act = np.argmax(self.model.predict(x[np.newaxis, ...]), axis=1)[0]
            probs[idx, act] += 1 - sum(probs[idx])

        return probs

    def __call__(self, states):
        return np.atleast_1d(self.sample(np.atleast_1d(states)))

