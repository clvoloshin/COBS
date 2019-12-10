import numpy as np
import scipy.signal as signal

class QWrapper(object):
    def __init__(self, Q, mapping, is_model=False, action_space_dim=2, modeltype='conv'):
        self.Q_values = Q
        self.map = mapping
        self.is_model = is_model
        self.action_space_dim = action_space_dim
        self.modeltype=modeltype

    def Q(self, policy, x, t=0):
        if not self.is_model:
            Qs = []
            for state in np.squeeze(x):
                Qs.append(self.Q_values[self.map[state]])
            return  np.array(Qs)
        else:
            if self.modeltype != 'linear':
                return self.Q_values.predict(x)
            else:
                inp = np.repeat(x, self.action_space_dim, axis=0)
                act = np.tile(np.arange(self.action_space_dim), len(x))
                inp = np.hstack([inp.reshape(inp.shape[0],-1), np.eye(self.action_space_dim)[act]])
                val= self.Q_values.predict(inp).reshape(-1, self.action_space_dim)
                return val



    def V(self, policy, x, t=0):
        if not self.is_model:
            return np.sum([self.Q_values[self.map[x],act]*prob for act,prob in enumerate(policy.predict([x])[0])])
        else:
            if self.action_space_dim is None:
                self.action_space_dim = len(policy.predict(x)[0])
            return np.sum([self.Q_values.predict([x,np.eye(self.action_space_dim)[[act]]])[0][0]*prob for act,prob in enumerate(policy.predict(x)[0])])
