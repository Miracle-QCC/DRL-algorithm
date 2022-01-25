import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from copy import deepcopy
import itertools
import core as core
import numpy as np
import random
import os
from torch.nn import functional as F
# from tensorboardX import SummaryWriter

# writer = SummaryWriter('runs/loss')
os.environ["CUDA_VISIBLE_DEVICES"] = "5，4"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SAC:
    def __init__(self, obs_dim, act_dim, act_bound, actor_critic=core.MLPActorCritic, seed=0,
                 replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=1e-4, alpha=0.3, auto_alpha=True):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_bound = act_bound
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = torch.tensor(alpha).float().to(device)
        self.capacity = replay_size
        self.time = 0
        self.target_entropy = torch.tensor(-act_dim).float()
        self.auto_alpha = auto_alpha
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.lossfn_q = nn.HuberLoss()

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
        self.alpha_optimizer = Adam([self.alpha] ,lr=lr)
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
            o = torch.FloatTensor(np.array(o)).to(device)
            a = torch.FloatTensor(np.array(a)).to(device)
            r = torch.FloatTensor(np.array(r)).to(device)
            o2 = torch.FloatTensor(np.array(o2)).to(device)
            d = torch.FloatTensor(np.array(d)).to(device)
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2,deterministic=True)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            #####   My modified location
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)
            a,_ = self.ac.pi(o)
        # Huber loss against Bellman backup
        loss_q1 = self.lossfn_q(q1,backup)
        loss_q2 = self.lossfn_q(q2,backup)
        consisitncy_pen = 0
        for i in range(8):
            sample_action = torch.rand(o.shape[0], self.act_dim).to(torch.float32).to(device) * 2 - 1
            consisitncy_pen += (F.relu(self.ac.q1(o,sample_action) - self.ac.q1(o,a)) \
                               + F.relu(self.ac.q2(o,sample_action) - self.ac.q2(o,a))) / 2.0
        consisitncy_pen /= 8.0
        loss_q = loss_q1 + loss_q2 + 0.25 * consisitncy_pen.mean()
        #writer.add_scalar('loss_q',loss_q,global_step=self.time)
        # Useful info for logging
        #q_info = dict(Q1Vals=q1.detach().numpy(),
                      #Q2Vals=q2.detach().numpy())

        return loss_q
    # Set up function for computing SAC_baseline pi loss
    def compute_loss_pi(self, data):
        o,_,_,_,_ = zip(*data)
        o = torch.FloatTensor(np.array(o)).to(device)
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        #####   My modified location
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        #writer.add_scalar('-loss_pi', -loss_pi, global_step=self.time)
        # Useful info for logging
        #pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi,logp_pi

    def compute_loss_alpha(self,log_p):
        loss_alpha = -self.alpha * (log_p + self.target_entropy).detach().mean()
        return loss_alpha

    def update(self, batch_size):
        self.time += 1
        # First run one gradient descent step for Q1 and Q2
        # delay update
        for i in range(2):
            data = random.sample(self.replay_buffer, batch_size)
            self.q_optimizer.zero_grad()
            loss_q = self.compute_loss_q(data)
            loss_q.backward()
            self.q_optimizer.step()
        data = random.sample(self.replay_buffer, batch_size)
        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi ,logp_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # update alpha
        if self.auto_alpha:
            self.alpha.requires_grad = True
            loss_alpha = self.compute_loss_alpha(log_p= logp_pi)
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
        self.alpha.requires_grad = False


        # Unfreeze Q-networks so you can optimize it at next DDPG_Adv step.
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
        o = torch.FloatTensor(o).view(1,-1).to(device)
        a = self.ac.act(o,deterministic).squeeze()
        return a
