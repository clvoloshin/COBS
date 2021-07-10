import torch
import torch.nn as nn

# Simulator, Model definition.
class defaultCNN(nn.Module):
    def __init__(self, shape, action_space_dim):
        super(defaultCNN, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(shape[0], 16, (2,2)),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16*(shape[1]-1)*(shape[2]-1), 8),
            nn.ELU(),
            nn.Linear(8, 8),
            nn.ELU(),
            nn.Linear(8, action_space_dim)
        )
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=.001)
            torch.nn.init.normal_(m.bias, mean=0.0, std=.001)

    def forward(self, state, action):
        output = self.net(state)
        return torch.masked_select(output, action)
    
    def predict(self, state):
        return self.net(state)
