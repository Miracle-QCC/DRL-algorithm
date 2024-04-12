import torch
from torch import nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_size,act_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim,hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size,act_dim)
        # self.fc3 = nn.Linear(hidden_size,act_dim)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.act_bound = act_bound
    def forward(self,obs, h=None):
        x = self.ReLU(self.fc1(obs))
        y,h = self.rnn(x,h)
        z = self.Tanh(self.fc2(y))
        return z * self.act_bound, h

class Critic(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim+act_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)
        # self.fc3 = nn.Linear(hidden_size,1)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.ReLU = nn.ReLU()

    def forward(self,obs,act,h=None):
        input = torch.cat([obs,act],-1)
        x = self.ReLU(self.fc1(input))
        # y = self.ReLU(self.fc2(x))
        y,h = self.rnn(x,h)
        z = self.fc2(y)
        return z,h

class DDActorCritic(nn.Module):
    def __init__(self,obs_dim,act_dim,act_bound,hidden_space):
        super().__init__()
        self.pi = Actor(obs_dim,act_dim,hidden_space,act_bound)
        self.q = Critic(obs_dim,act_dim,hidden_space)
    def act(self,obs, h):
        action,h = self.pi(obs, h)
        return action.detach().cpu().numpy(),h