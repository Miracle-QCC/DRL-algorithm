import torch
from torch.optim import Adam
from copy import deepcopy
import itertools
import core as core
import numpy as np
import random
#from tensorboardX import SummaryWriter

#writer = SummaryWriter('runs/loss')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SAC:
    def __init__(self, obs_dim, act_dim, act_bound, actor_critic=core.MLPActorCritic, seed=0,
                 replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.3):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.capacity = replay_size
        self.time = 0
        act_bound = torch.FloatTensor(act_bound).to(device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


        self.ac = actor_critic(obs_dim, act_dim, act_limit=act_bound).to(device)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Experience buffer
        self.replay_buffer = []

    def store(self,*sample):
        if len(self.replay_buffer) == self.capacity:
            self.replay_buffer.remove(self.replay_buffer[0])
        self.replay_buffer.append(sample)

    # def sample_batch(self,batch_size):
    #     batch = random.sample(self.replay_buffer,batch_size)
    #     return batch

    # Set up function for computing SAC_baseline Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = zip(*data)
        with torch.no_grad():
            o = torch.FloatTensor(o).to(device)
            a = torch.FloatTensor(a).to(device)
            r = torch.FloatTensor(r).to(device)
            o2 = torch.FloatTensor(o2).to(device)
            d = torch.FloatTensor(d).to(device)
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            #####   My modified location
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2 / (torch.tanh(q_pi_targ) + 2))

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        #writer.add_scalar('loss_q',loss_q,global_step=self.time)
        # Useful info for logging
        #q_info = dict(Q1Vals=q1.detach().numpy(),
                      #Q2Vals=q2.detach().numpy())

        return loss_q
    # Set up function for computing SAC_baseline pi loss
    def compute_loss_pi(self, data):
        o,_,_,_,_ = zip(*data)
        o = torch.FloatTensor(o).to(device)
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        #####   My modified location
        loss_pi = (self.alpha * logp_pi / ((torch.tanh(q_pi) + 2)) - q_pi).mean()
        #writer.add_scalar('-loss_pi', -loss_pi, global_step=self.time)
        # Useful info for logging
        #pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi
    def update(self, batch_size):
        self.time += 1
        # First run one gradient descent step for Q1 and Q2
        data = random.sample(self.replay_buffer, batch_size)
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        o = torch.FloatTensor(o).to(device)
        return self.ac.act(o,deterministic)