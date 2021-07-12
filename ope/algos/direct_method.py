from tqdm import trange
import numpy as np
import scipy.signal as signal

class DirectMethod(object):
    """Direct Method Base Class.
    """
    def __init__(self) -> None:
        self.fitted = None
    
    def fit_tabular(self, behavior_data, pi_e, cfg) -> None:
        NotImplemented

    def fit_NN(self, behavior_data, pi_e, cfg) -> None:
        NotImplemented
    
    def fit(self, behavior_data, pi_e, cfg, modeltype) -> None:
        if modeltype == 'tabular':
            self.fit_tabular(behavior_data, pi_e, cfg)
        else:
            self.fit_NN(behavior_data, pi_e, cfg)
    
class DirectMethodQ(DirectMethod):
    """Direct Method Q Abstract Base Class.

    These are Direct Methods that produce Q functions
    """
    def __init__(self) -> None:
        DirectMethod.__init__(self)
        self.fitted = None
    
    def Q_tabular(self, states, actions=None) -> np.ndarray:
        NotImplemented
    
    def Q_NN(self, states, actions=None) -> np.ndarray:
        NotImplemented
    
    def Q(self, states, actions=None) -> np.ndarray:
        if self.fitted is None:
            raise 'Need to call "fit" before using this method'
        elif self.fitted == 'tabular':
            return self.Q_tabular(states, actions)
        else:
            return self.Q_NN(states, actions)
    
    def predict(self, states, actions=None) -> np.ndarray:
        if self.fitted is None:
            raise 'Need to call "fit" before using this method'
        elif self.fitted == 'tabular':
            return self.Q_tabular(states, actions)
        else:
            return self.Q_NN(states, actions)
    
    def get_Qs_for_data(self, data, cfg) -> list:
        Qs = []
        batchsize = 1
        num_batches = int(np.ceil(len(data)/batchsize))
        
        for batchnum in trange(num_batches, desc='Batch'):
            low_ = batchsize*batchnum
            high_ = min(batchsize*(batchnum+1), len(data))

            pos = data.states(False, low_=low_,high_=high_)
            acts = data.actions()[low_:high_]

            traj_Qs = self.Q(cfg.processor(pos))

            traj_Qs = traj_Qs.reshape(-1, data.n_actions)
        
            Qs.append(traj_Qs)

        return Qs

class DirectMethodWeight(DirectMethod):
    """Direct Method Weight Abstract Base Class.

    These are Direct Methods that produce weight functions
    """
    def __init__(self) -> None:
        DirectMethod.__init__(self)
        self.fitted = None
    
    def evaluate_NN(self, data, cfg):
        NotImplemented
    
    def evaluate_tabular(self, data, cfg):
        NotImplemented
    
    def evaluate(self, data, cfg) -> float:
        if self.fitted is None:
            raise 'Need to call "fit" before using this method'
        elif self.fitted == 'tabular':
            return self.evaluate_tabular(data, cfg)
        else:
            return self.evaluate_NN(data, cfg)

class DirectMethodModelBased(DirectMethod):
    """Direct Method Model Based Abstract Base Class.

    These are Direct Methods that are Model Based
    """
    def __init__(self) -> None:
        DirectMethod.__init__(self)
        self.fitted = None
    
    def Q_tabular(self, policy, state, gamma) -> np.ndarray:
        NotImplemented
    
    def Q_NN(self, policy, state, gamma) -> np.ndarray:
        NotImplemented
    
    def Q(self, policy, state, gamma) -> np.ndarray:
        if self.fitted is None:
            raise 'Need to call "fit" before using this method'
        elif self.fitted == 'tabular':
            return self.Q_tabular(policy, state, gamma)
        else:
            return self.Q_NN(policy, state, gamma)
    
    def predict(self, policy, state, gamma) -> np.ndarray:
        if self.fitted is None:
            raise 'Need to call "fit" before using this method'
        elif self.fitted == 'tabular':
            return self.Q_tabular(policy, state, gamma)
        else:
            return self.Q_NN(policy, state, gamma)
    
    def get_Qs_for_data(self, policy, data, cfg) -> list:
        Qs = []
        batchsize = 1
        num_batches = int(np.ceil(len(data)/batchsize))
        
        for batchnum in trange(num_batches, desc='Batch'):
            low_ = batchsize*batchnum
            high_ = min(batchsize*(batchnum+1), len(data))

            pos = data.states(False, low_=low_,high_=high_)
            acts = data.actions()[low_:high_]

            traj_Qs = self.Q(policy, cfg.processor(pos), cfg.gamma)

            traj_Qs = traj_Qs.reshape(-1, data.n_actions)
        
            Qs.append(traj_Qs)

        return Qs
    