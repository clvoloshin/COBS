import numpy as np


class TabularPolicy(object):
    def __init__(self, dic, actions=4, absorbing=None):
        self.dic = dic
        self.actions = actions
        self.absorbing = absorbing

    def predict(self, xs, **kw):
        if (len(xs.shape) == 3) and (xs.shape[-1] == xs.shape[-2]) and (xs.shape[-2] != 1):
            import pdb; pdb.set_trace()
            if np.sum(xs) == 0:
                xs = np.array([self.absorbing])
            else:
                xs = xs.reshape(-1)
        out = []
        xs = xs.reshape(-1)
        for x in xs:
            probs = np.zeros(self.actions)
            if x in self.dic:
            	probs[self.dic[x]] = 1.
            elif x == self.absorbing[0]:
            	probs[0] = 1.
            else:
            	raise
            out.append(probs)
        return np.array(out)

    def sample(self, xs):
        return self(xs)

    def __call__(self, states):
        out = []
        for x in states:
            out.append(self.dic[x])
        return np.array(out)
