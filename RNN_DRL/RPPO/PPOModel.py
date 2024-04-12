import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## RPPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,is_continous,hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size,action_dim)

        self.ReLU = nn.ReLU()
        if is_continous:
            self.last_act = None
        else:
            self.last_act = nn.Softmax(dim=-1)

    def forward(self,x,h=None):
        x = self.ReLU(self.fc1(x))
        x,h = self.rnn(x,h)
        if self.last_act:
            out = self.last_act(self.fc2(x))
        else:
            out = self.fc2(x)
        return out,h

class Critic(nn.Module):
    def __init__(self,obs_dim,act_dim,hidden_size=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim,hidden_size)
        # self.fc3 = nn.Linear(hidden_size,1)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size,1)
        self.ReLU = nn.ReLU()

    def forward(self,obs,h=None):
        x = self.ReLU(self.fc1(obs))
        y,h = self.rnn(x,h)
        z = self.fc2(y)
        return z,h

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, hidden_space=128):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        self.actor = Actor(state_dim,action_dim,has_continuous_action_space, hidden_space)
        # critic
        self.critic = Critic(state_dim,action_dim,hidden_space)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]).to(device), torch.zeros([1, batch_size, self.hidden_space]).to(device)
        else:
            return torch.zeros([1, 1, self.hidden_space]).to(device), torch.zeros([1, 1, self.hidden_space]).to(device)

    def act(self, state, deter=False, h_a=None,h_c=None):

        if self.has_continuous_action_space:
            action_mean,h_a = self.actor(state,h_a)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs,h_a = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val,h_c = self.critic(state,h_c)
        if deter:
            try:
                action = action_mean
            except:
                action = torch.argmax(action_probs)
        return action.detach(), action_logprob.detach(), state_val.detach(), h_a, h_c

    def evaluate(self, state, action):
        b = len(state)
        if self.has_continuous_action_space:
            action_mean, _ = self.actor(state)

            action_var = self.action_var.expand_as(action_mean).reshape(b,-1,self.action_dim)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

        else:
            action_probs, _ = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values, _ = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=5e-4, lr_critic=1e-3, gamma=0.99, K_epochs=80, eps_clip=0.3,
                 has_continuous_action_space=True, action_std_init=0.6, hidden_space=128):

        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        if has_continuous_action_space:
            self.action_std = action_std_init
        self.hidden_space = hidden_space
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]).to(device), torch.zeros([1, batch_size, self.hidden_space]).to(device)
        else:
            return torch.zeros([1, 1, self.hidden_space]).to(device), torch.zeros([1, 1, self.hidden_space]).to(device)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling RPPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling RPPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def get_action(self, state, deter=False, h_a=None, h_c=None):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device).unsqueeze(0).unsqueeze(0)
                action, action_logprob, state_val, h_a,h_c = self.policy_old.act(state, deter, h_a,h_c)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten(), h_a, h_c
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device).unsqueeze(0).unsqueeze(0)
                action, action_logprob, state_val, h_a,h_c = self.policy_old.act(state, deter, h_a,h_c)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item(), h_a, h_c

    def get_rnn_sample(self, s,a,r,adv,logp,rewards, look_setup=20, batch_size = 128):
        max_seq_len = len(s)
        idxs = np.random.choice(list(range(max_seq_len - look_setup)),size=batch_size,replace=False)
        new_s = []
        new_a = []
        new_r = []
        new_adv = []
        new_logp = []
        new_rew = []
        for idx in idxs:
            new_s.append(s[idx:idx+look_setup])
            new_a.append(a[idx:idx+look_setup])
            new_r.append(r[idx:idx+look_setup])
            new_adv.append(adv[idx:idx+look_setup])
            new_logp.append(logp[idx:idx+look_setup])
            new_rew.append(rewards[idx:idx+look_setup])
        new_s = torch.stack(new_s).reshape(batch_size,look_setup,-1)
        new_a = torch.stack(new_a).reshape(batch_size,look_setup,-1)
        new_r = torch.stack(new_r).reshape(batch_size,look_setup,-1)
        new_adv = torch.stack(new_adv).reshape(batch_size,look_setup,-1)
        new_logp = torch.stack(new_logp).reshape(batch_size,look_setup,-1)
        new_rew = torch.stack(new_rew).reshape(batch_size,look_setup,-1)

        return new_s,new_a,new_r,new_adv,new_logp,new_rew

    def update(self, look_setup=20, batch_size=128):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device).unsqueeze(0)
        if self.has_continuous_action_space:
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device).unsqueeze(0).reshape(1,-1,self.action_dim)
        else:
            old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device).unsqueeze(0).reshape(1,-1,1)

        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device).unsqueeze(0)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device).unsqueeze(0)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        # old_states,old_actions,old_state_values,advantages,old_logprobs,rewards = \
        #     self.get_rnn_sample(old_states,old_actions,old_state_values,advantages,old_logprobs,rewards,look_setup,batch_size)
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            # state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective RPPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), rewards) - 0.01 * dist_entropy.unsqueeze(-1)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=2)
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(),max_norm=4)
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


