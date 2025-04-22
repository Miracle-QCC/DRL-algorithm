import pickle
import scipy.signal
from dm_control import suite, viewer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm


class policy_net(nn.Module):
    def __init__(self, xdim, udim, hdim=128):
        super().__init__()

        self.xdim = xdim
        self.udim = udim
        self.fc1 = nn.Linear(xdim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, hdim)
        self.fc_mu = nn.Linear(hdim, udim)
        self.fc_log_std = nn.Linear(hdim, udim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std)
        return mu, std


class value_net(nn.Module):
    def __init__(self, xdim, hdim=128):
        super().__init__()

        self.fc1 = nn.Linear(xdim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, hdim)
        self.fc_value = nn.Linear(hdim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        value = self.fc_value(x)
        return value


class PPO:
    def __init__(self, xdim, udim, hdim=32, actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, lambd=0.97, K_epochs=6, eps_clip=0.2):

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # actor and critic are on GPU
        self.actor = policy_net(xdim, udim, hdim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)

        self.critic = value_net(xdim, hdim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lambd = lambd
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip

    # input: numpy 0d state, CPU
    # output: numpy 0d action, CPU
    def select_action(self, state):
        # add a dimension, to 1 x xdim, tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # forward pass through policy network
        mu, std = self.actor(state)

        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()

        return action.detach().cpu().numpy()[0]

    def update(self, buffer):

        def compute_advantage(x, discount):
            # Convert tensor to numpy, move to CPU
            x = x.detach().cpu().numpy()
            result = scipy.signal.lfilter(
                [1], [1, float(-discount)], x[::-1], axis=0)[::-1]
            # Convert back to tensor
            return torch.tensor(result.copy(), dtype=torch.float).to(self.device)

        s = torch.tensor(
            np.array(buffer['x']), dtype=torch.float).to(self.device)
        a = torch.tensor(
            np.array(buffer['u']), dtype=torch.float).to(self.device)
        r = torch.tensor(
            np.array(buffer['r']), dtype=torch.float).view(-1, 1).to(self.device)
        next_s = torch.tensor(
            np.array(buffer['next_x']), dtype=torch.float).to(self.device)
        done = torch.tensor(
            np.array(buffer['done']), dtype=torch.float).view(-1, 1).to(self.device)

        TD = r + self.gamma * self.critic(next_s) * (1 - done)
        delta = TD - self.critic(s)
        # GAE advantage
        advantage = compute_advantage(
            delta, self.gamma * self.lambd)  # Compute advantage on CPU

        assert advantage.shape == done.shape
        mu, std = self.actor(s)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(a)

        for _ in range(self.K_epochs):
            mu, std = self.actor(s)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(a)
            ratio = torch.exp(log_probs - old_log_probs)
            option1 = ratio * advantage
            option2 = torch.clamp(ratio, 1 - self.eps_clip,
                                  1 + self.eps_clip) * advantage
            actor_loss = -torch.min(option1, option2).mean()
            critic_loss = F.mse_loss(self.critic(s), TD.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()
