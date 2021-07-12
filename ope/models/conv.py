import torch
import torch.nn as nn
import numpy as np

class defaultCNN(nn.Module):
    def __init__(self, shape, action_space_dim):
        super(defaultCNN, self).__init__()
        self.c, self.h, self.w = shape

        self.net = nn.Sequential(
            nn.Conv2d(self.c, 16, (2,2)),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16*(self.h-1)*(self.w-1), 8),
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
    
    def predict_w_softmax(self, state):
        return nn.Softmax()(self.net(state))

class defaultModelBasedCNN(nn.Module):
    def __init__(self, shape, action_space_dim):
        super(defaultModelBasedCNN, self).__init__()
        self.c, self.h, self.w = shape

        self.features = nn.Sequential(
            nn.Conv2d(self.c, 4, (5, 5)),
            nn.ELU(),
            nn.Conv2d(4, 8, (3, 3)),
        )

        self.states_head = nn.Sequential(
            nn.ConvTranspose2d(8, 16, (3, 3)),
            nn.ELU(),
            nn.ConvTranspose2d(16, action_space_dim, (5, 5)),
        )
        
        self.rewards_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*(self.h-4-2)*(self.w-4-2), 8),
            nn.ELU(),
            nn.Linear(8, action_space_dim),
        )

        self.dones_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*(self.h-4-2)*(self.w-4-2), 8),
            nn.ELU(),
            nn.Linear(8, action_space_dim),
            nn.Sigmoid()
        )
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=.001)
            torch.nn.init.normal_(m.bias, mean=0.0, std=.001)

    def forward(self, state, action):
        T, R, D = self.states_head(self.features(state)), self.rewards_head(self.features(state)), self.dones_head(self.features(state))
        return T[np.arange(len(action)), action.float().argmax(1), ...][:,None,:,:], torch.masked_select(R, action), torch.masked_select(D, action)
    
    def predict(self, state):
        return self.states_head(self.features(state)), self.rewards_head(self.features(state)), self.dones_head(self.features(state))