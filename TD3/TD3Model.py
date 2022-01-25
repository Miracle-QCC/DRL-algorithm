from models import Actor_Critic
import torch
from torch.optim import Adam
from copy import deepcopy
import itertools
import random
import numpy as np
import torch.nn as nn

targ_noise = 0.2
noise_bound = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class TD3:
    def __init__(self,obs_dim,act_dim,act_bound,policy_delay=2,replay_size=int(1e5),lr=1e-3,gamma=0.99,polyak=0.98,seed=1):
        self.capacity = replay_size
        self.gamma = gamma
        self.polyka = polyak
        self.policy_delay = policy_delay # Interval of update policies
        self.time = 0
        self.experience = []
        self.act_bound = act_bound[0]
        self.act_dim = act_dim
        torch.manual_seed(seed)
        random.seed(seed)

        act_bound = torch.FloatTensor(act_bound).to(device)
        self.ac = Actor_Critic(obs_dim,act_dim,act_bound).to(device)
        self.ac_targ = deepcopy(self.ac)

        for p in self.ac_targ.parameters():
            p.requires_grad = False

        self.q_param = itertools.chain(self.ac.q1.parameters(),self.ac.q2.parameters())

        self.q_optimizer = Adam(self.q_param,lr)
        self.pi_optimizer = Adam(self.ac.pi.parameters(),lr)

    def get_action(self,obs):
        obs = torch.FloatTensor(obs).to(device)
        action = self.ac.act(obs)
        # get the noise
        act_noise = np.random.rand(self.act_dim) * targ_noise
        action = action + act_noise

        action = np.clip(action,-self.act_bound,self.act_bound)
        return action

    def store(self,*sample):
        if len(self.experience) == self.capacity:
            self.experience.pop(0)
        self.experience.append(sample)

    def update(self,batch_size):
        if len(self.experience) < batch_size:
            return
        self.time += 1   # Training times plus one
        sample = random.sample(self.experience,batch_size)
        s,a,r,s2,d = zip(*sample)
        with torch.no_grad():
            s = torch.FloatTensor(s).to(device)
            a = torch.FloatTensor(a).to(device)
            r = torch.FloatTensor(r).to(device).view(batch_size,-1)
            s2 = torch.FloatTensor(s2).to(device)
            d = torch.FloatTensor(d).to(device).view(batch_size,-1)

        # update the critic network
        q1_pi = self.ac.q1(s,a)
        q2_pi = self.ac.q2(s,a)
        with torch.no_grad():
            pi_targ = self.ac_targ.pi(s2)
            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * targ_noise
            epsilon = torch.clamp(epsilon, -noise_bound, noise_bound)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.act_bound, self.act_bound)
            q1_targ = self.ac_targ.q1(s2,a2)
            q2_targ = self.ac_targ.q2(s2,a2)
            q_targ = torch.min(q1_targ,q2_targ)
            backup = r + self.gamma * (1-d) * q_targ
        loss_fn = nn.MSELoss()
        loss_q1 = loss_fn(q1_pi,backup)
        loss_q2 = loss_fn(q2_pi,backup)
        loss_q = loss_q1 + loss_q2
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # update the actor network

        if self.time % self.policy_delay == 0:
            pi_action = self.ac.pi(s)
            loss_pi = -torch.mean(self.ac.q1(s,pi_action))
            for p in self.q_param:
                p.requires_grad = False

            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()

            for p in self.q_param:
                p.requires_grad = True

            with torch.no_grad():
                for p,p_targ in zip(self.ac.parameters(),self.ac_targ.parameters()):
                    p_targ.data.copy_(p_targ.data * self.polyka + p.data * (1 - self.polyka))
