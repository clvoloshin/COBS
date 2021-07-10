import numpy as np

class DirectMethod(object):
    """Direct Method Base Class.
    """
    def __init__(self) -> None:
        self.fitted = None

    def fit_NN(self) -> None:
        NotImplemented
    
    def fit_tabular(self) -> None:
        NotImplemented
    
    def Q_NN(self, states, actions=None) -> np.ndarray:
        NotImplemented
    
    def Q_tabular(self, states, actions=None) -> np.ndarray:
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
    
