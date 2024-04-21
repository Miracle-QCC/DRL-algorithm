import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class DisActor(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_size):
        super(DisActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,act_dim)
        self.ReLU = nn.ReLU()
        self.sg = nn.Softmax(dim=-1)

    def forward(self,obs):
        x = self.ReLU(self.fc1(obs))
        y = self.ReLU(self.fc2(x))
        z = self.sg(self.fc3(y))
        return z

class Critic(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim+act_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,1)
        self.ReLU = nn.ReLU()

    def forward(self,obs,act):
        input = torch.cat([obs,act],1)
        x = self.ReLU(self.fc1(input))
        y = self.ReLU(self.fc2(x))
        z = self.fc3(y)
        return z

class DDActorCritic(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super().__init__()
        self.pi = DisActor(obs_dim,act_dim,256)
        self.q = Critic(obs_dim,act_dim,256)
    def act(self,obs,deter):
        prob = self.pi(obs).detach().cpu().numpy()
        if deter:
            action = np.argmax(prob)
        else:
            noise = np.random.gumbel(size=len(prob))
            action = np.argmax(prob + noise)
        return action