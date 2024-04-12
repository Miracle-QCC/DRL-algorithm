import collections
import sys

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

class EpisodeBuffer:
    """A simple numpy replay buffer."""

    def __init__(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []

    def put(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.next_obs.append(transition[3])
        self.done.append(transition[4])

    def sample(self, random_update=False, lookup_step=None, idx=None):
        obs = np.array(self.obs)
        action = np.array(self.action)
        reward = np.array(self.reward)
        next_obs = np.array(self.next_obs)
        done = np.array(self.done)

        if random_update is True:
            obs = obs[idx:idx + lookup_step]
            action = action[idx:idx + lookup_step]
            reward = reward[idx:idx + lookup_step]
            next_obs = next_obs[idx:idx + lookup_step]
            done = done[idx:idx + lookup_step]

        return dict(obs=obs,
                    acts=action,
                    rews=reward,
                    next_obs=next_obs,
                    done=done)

    def __len__(self) -> int:
        return len(self.obs)


class EpisodeMemory():
    """Episode memory for recurrent agent"""

    def __init__(self, random_update=False,
                 max_epi_num=100, max_epi_len=500,
                 batch_size=1,
                 lookup_step=None):
        self.random_update = random_update  # if False, sequential update
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.batch_size = batch_size
        self.lookup_step = lookup_step

        if (random_update is False) and (self.batch_size > 1):
            sys.exit(
                'It is recommend to use 1 batch for sequential update, if you want, erase this code block and modify code')

        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, episode):
        self.memory.append(episode)

    def sample(self):
        sampled_buffer = []

        ##################### RANDOM UPDATE ############################
        if self.random_update:  # Random upodate
            sampled_episodes = random.sample(self.memory, self.batch_size)

            check_flag = True  # check if every sample data to train is larger than batch size
            min_step = self.max_epi_len

            for episode in sampled_episodes:
                min_step = min(min_step, len(episode))  # get minimum step from sampled episodes

            for episode in sampled_episodes:
                if min_step > self.lookup_step:  # sample buffer with lookup_step size
                    idx = np.random.randint(0, len(episode) - self.lookup_step + 1)
                    sample = episode.sample(random_update=self.random_update, lookup_step=self.lookup_step, idx=idx)
                    sampled_buffer.append(sample)
                else:
                    idx = np.random.randint(0, len(episode) - min_step + 1)  # sample buffer with minstep size
                    sample = episode.sample(random_update=self.random_update, lookup_step=min_step, idx=idx)
                    sampled_buffer.append(sample)

        ##################### SEQUENTIAL UPDATE ############################
        else:  # Sequential update
            idx = np.random.randint(0, len(self.memory))
            sampled_buffer.append(self.memory[idx].sample(random_update=self.random_update))

        return sampled_buffer, len(sampled_buffer[0]['obs'])  # buffers, sequence_length

    def __len__(self):
        return len(self.memory)


class Agent(object):
    def __init__(self,obs_dim,act_dim,act_bound,capacity=int(1e5),random_update=True,batch_size=64,
                 max_epi_num=100, max_epi_len=600,lookup_step=20,gamma=0.99,lr=1e-3,tau=0.98,seed = 1,hidden_space=256):
        self.gamma = gamma
        self.tau = tau
        self.capacity = capacity
        self.experience = EpisodeMemory(random_update=random_update,
                                   max_epi_num=max_epi_num, max_epi_len=max_epi_len,
                                   batch_size=batch_size,
                                   lookup_step=lookup_step)
        torch.manual_seed(seed)
        random.seed(seed)

        self.act_bound = act_bound[0]
        act_bound = torch.FloatTensor(act_bound).to(device)
        self.act_dim = act_dim
        self.hidden_space = hidden_space
        self.ac = DDActorCritic(obs_dim,act_dim,act_bound,hidden_space=hidden_space).to(device)
        self.ac_targ = deepcopy(self.ac)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.pi_optimizer = Adam(self.ac.pi.parameters(),lr)
        self.q_optimizer = Adam(self.ac.q.parameters(),lr)

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]).to(device), torch.zeros([1, batch_size, self.hidden_space]).to(device)
        else:
            return torch.zeros([1, 1, self.hidden_space]).to(device), torch.zeros([1, 1, self.hidden_space]).to(device)

    def get_action(self,obs, h, deter=False):
        obs = torch.FloatTensor(obs).to(device).unsqueeze(0).unsqueeze(0)
        h = h.to(device)
        mean,h = self.ac.act(obs,h)
        if deter:
            action = mean
        else:
            action = mean + targ_noise * np.random.rand(self.act_dim)
            action = np.clip(action,-self.act_bound,self.act_bound)
        return action,h

    def store(self,episode_data):
        self.experience.put(episode_data)

    def update(self , batch_size):
        if len(self.experience) < 2 * batch_size:
            return
        # sample = random.sample(self.experience,batch_size)
        # s,a,r,s2,d = zip(*sample)
        samples, seq_len = self.experience.sample()

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for i in range(batch_size):
            observations.append(samples[i]["obs"])
            actions.append(samples[i]["acts"])
            rewards.append(samples[i]["rews"])
            next_observations.append(samples[i]["next_obs"])
            dones.append(samples[i]["done"])
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        dones = np.array(dones)

        s = torch.FloatTensor(observations.reshape(batch_size, seq_len, -1)).to(device)
        a = torch.LongTensor(actions.reshape(batch_size, seq_len, -1)).to(device)
        r = torch.FloatTensor(rewards.reshape(batch_size, seq_len, -1)).to(device)
        s2 = torch.FloatTensor(next_observations.reshape(batch_size, seq_len, -1)).to(device)
        d = torch.FloatTensor(dones.reshape(batch_size, seq_len, -1)).to(device)

        h_a,h_c = self.init_hidden_state(batch_size=batch_size,training=True)
        q_pi,h_c = self.ac.q(s,a,h_c)
        with torch.no_grad():
            h_a_t, h_c_t = self.init_hidden_state(batch_size=batch_size, training=True)
            a2,h_a_t = self.ac_targ.pi(s2,h_a_t)
            epsilon = torch.randn_like(a2) * targ_noise
            epsilon = torch.clamp(epsilon, -noise_bound, noise_bound)
            a2 = a2 + epsilon
            a2 = torch.clamp(a2, -self.act_bound, self.act_bound)
            q_targ,h_c_t = self.ac_targ.q(s2,a2,h_c_t)
            backup = r + self.gamma * q_targ * (1 - d)
        loss_fn = nn.MSELoss()
        loss_q = loss_fn(q_pi , backup)
        # return loss_q
        # #####First update the value network
        # loss_q = compute_loss_q()
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()


        pi_action,h_a = self.ac.pi(s,h_a)
        loss_pi = -torch.mean(self.ac.q(s,pi_action)[0])

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

