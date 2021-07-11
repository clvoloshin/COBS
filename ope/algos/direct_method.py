import numpy as np

class DirectMethod(object):
    """Direct Method Base Class.
    """
    def __init__(self) -> None:
        self.fitted = None
    
    def fit_tabular(self, behavior_data, pi_e, cfg) -> None:
        NotImplemented

    def Q_tabular(self, states, actions=None) -> np.ndarray:
        NotImplemented
    
    def fit_NN(self, behavior_data, pi_e, cfg) -> None:
        NotImplemented
    
    def Q_NN(self, states, actions=None) -> np.ndarray:
        NotImplemented
    
    def fit(self, behavior_data, pi_e, cfg, modeltype) -> None:
        if modeltype == 'tabular':
            self.fit_tabular(behavior_data, pi_e, cfg)
        else:
            self.fit_NN(behavior_data, pi_e, cfg)
    
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
    
