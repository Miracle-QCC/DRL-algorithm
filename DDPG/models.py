import torch
from torch import nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_size,act_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,act_dim)
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.act_bound = act_bound
    def forward(self,obs):
        x = self.ReLU(self.fc1(obs))
        y = self.ReLU(self.fc2(x))
        z = self.Tanh(self.fc3(y))
        return z * self.act_bound

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
    def __init__(self,obs_dim,act_dim,act_bound):
        super().__init__()
        self.pi = Actor(obs_dim,act_dim,256,act_bound)
        self.q = Critic(obs_dim,act_dim,256)
    def act(self,obs):
        action = self.pi(obs)
        return action.detach().cpu().numpy()