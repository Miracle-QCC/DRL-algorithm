from torch.optim import Adam
from copy import deepcopy
import random
from models import DDActorCritic
import torch
from torch import nn
import numpy as np

targ_noise = 0.2
noise_bound = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Agent(object):
    def __init__(self,obs_dim,act_dim,capacity=int(1e5),gamma=0.99,lr=1e-3,tau=0.98,seed = 1):
        self.gamma = gamma
        self.tau = tau
        self.capacity = capacity
        self.experience = []
        torch.manual_seed(seed)
        random.seed(seed)

        self.act_dim = act_dim

        self.ac = DDActorCritic(obs_dim,act_dim).to(device)
        self.act_dim = act_dim
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.pi_optimizer = Adam(self.ac.pi.parameters(),lr)
        self.q_optimizer = Adam(self.ac.q.parameters(),lr)

    def get_action(self,obs,deter=False):
        obs = torch.FloatTensor(obs).to(device)
        # action = self.ac.act(obs) + targ_noise * np.random.rand(self.act_dim)
        # action = np.clip(action,-self.act_bound,self.act_bound)
        return self.ac.act(obs,deter)

    def store(self,*sample):
        if len(self.experience) == self.capacity:
            self.experience.pop(0)
        self.experience.append(sample)

    def get_one_hot_act(self,probs,batch_size,train=False):
        noise = torch.from_numpy(np.random.gumbel(size=(batch_size,self.act_dim))).to(device)
        if train:
            action = torch.argmax(probs + noise,dim=-1)
            a_onehot = torch.zeros((batch_size, self.act_dim)).scatter_(1, action.reshape(-1, 1), 1)
            a_onehot = (a_onehot - probs).detach() + probs
        else:
            action = torch.argmax(probs + noise,dim=-1)
            a_onehot = torch.zeros((batch_size, self.act_dim)).scatter_(1, action.reshape(-1, 1), 1)
        return a_onehot

    def update(self , batch_size):
        if len(self.experience) < 2 * batch_size:
            return
        sample = random.sample(self.experience,batch_size)
        s,a,r,s2,d = zip(*sample)

        s = torch.FloatTensor(s).to(device)
        a = torch.FloatTensor(a).to(device).long()
        a_one_hot = torch.zeros((batch_size,self.act_dim)).scatter_(1,a.reshape(-1,1),1)
        r = torch.FloatTensor(r).view(batch_size,-1).to(device)
        s2 = torch.FloatTensor(s2).to(device)
        d = torch.FloatTensor(d).view(batch_size,-1).to(device)


        q_pi = self.ac.q(s,a_one_hot)
        with torch.no_grad():
            a2_probs = self.ac_targ.pi(s2)
            a2_one_hot = self.get_one_hot_act(probs=a2_probs,batch_size=batch_size)
            # epsilon = torch.randn_like(a2) * targ_noise
            # epsilon = torch.clamp(epsilon, -noise_bound, noise_bound)
            # a2 = a2 + epsilon
            # a2 = torch.clamp(a2, -self.act_bound, self.act_bound)
            q_targ = self.ac_targ.q(s2,a2_one_hot)
            backup = r + self.gamma * q_targ * (1 - d)
        loss_fn = nn.MSELoss()
        loss_q = loss_fn(q_pi , backup)
        # return loss_q
        # #####First update the value network
        # loss_q = compute_loss_q()
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()


        pi_probs = self.ac.pi(s)
        pi_one_hot = self.get_one_hot_act(probs=pi_probs, batch_size=batch_size, train=True)
        loss_pi = -torch.mean(self.ac.q(s,pi_one_hot))

        #####Then update the policy network
        # for p in self.ac.q.parameters():
        #     p.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()
        # for p in self.ac.q.parameters():
        #     p.requires_grad = True

        #####soft update
        with torch.no_grad():
            for p,p_targ in zip(self.ac.parameters(),self.ac_targ.parameters()):
                p_targ.data.copy_(p_targ.data * self.tau + (1 - self.tau) * p.data)