# adapted from https://github.com/nikhilbarhate99/TD3-PyTorch-BipedalWalker-v2/blob/master/TD3.py

import torch
from torch import nn,optim
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    f_in = layer.weight.data.size(0)
    lim = 1. / np.sqrt(f_in)
    return (-lim,lim)

class Actor(nn.Module):
    def __init__(self,state_size,action_size,seed,max_action):
        super(Actor,self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.max_action = max_action
        fc1_units = 256
        fc2_units = 128
        
        self.fc1 = nn.Linear(state_size,fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,action_size)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return  torch.tanh(self.fc3(x)) * self.max_action
    


class Critic(nn.Module):
    def __init__(self,state_size,action_size,seed):
        super(Critic,self).__init__()
        
        self.seed = torch.manual_seed(seed)
        fc1_units = 256
        fc2_units = 128
        
        self.fc1 = nn.Linear(state_size+action_size,fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3,3e-3)
    
    def forward(self,state,action):
        xs = torch.cat((state,action),dim=1)
        x = F.relu(self.fc1(xs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        