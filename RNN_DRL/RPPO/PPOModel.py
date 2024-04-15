import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import Categorical
import numpy as np
import scipy
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
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


################################## PPO Policy ##################################

class RolloutBuffer:
    def __init__(self, gamma=0.99, lamb=0.95):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.gamma = gamma
        self.lamb = lamb

    def store(self, obs, act, ret, s_v, logp):
        self.states.append(obs)
        self.actions.append(act)
        self.rewards.append(ret)
        self.state_values.append(s_v)
        self.logprobs.append(logp)
    def __len__(self):
        return len(self.states)
    def finish_path(self):
        rewards = np.append(np.array(self.rewards), 0)
        state_values = np.append(np.array(self.state_values), 0)

        deltas = rewards[:-1] + self.gamma * state_values[1:] - state_values[:-1]
        self.adv_buf = discount_cumsum(deltas, self.gamma * self.lamb).tolist()

        # the next line computes rewards-to-go, to be targets for the value function
        self.rewards = discount_cumsum(rewards, self.gamma)[:-1].tolist()


class PPOBuffer:
    def __init__(self, gamma=0.99, lamb=0.95):
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def clear(self):
        del self.memory[:]

    def append(self, rooler):
        self.memory.append(rooler)

    def sample(self, lookup_step=20):
        obs = []
        act = []
        ret = []
        adv = []
        logp = []
        bs = len(self.memory)
        min_look = float('inf')
        for rooler in self.memory:
            min_look = min(min(len(rooler), lookup_step),min_look)

        for rooler in self.memory:
            if min_look > lookup_step:  # sample buffer with lookup_step size
                idx = np.random.randint(0, min_look - lookup_step + 1)
                obs.append(rooler.states[idx:idx+min_look])
                act.append(rooler.actions[idx:idx+min_look])
                ret.append(rooler.rewards[idx:idx+min_look])
                adv.append(rooler.adv_buf[idx:idx+min_look])
                logp.append(rooler.logprobs[idx:idx+min_look])
            else:
                idx = np.random.randint(0, len(rooler) - min_look + 1)
                obs.append(rooler.states[idx:idx + min_look])
                act.append(rooler.actions[idx:idx + min_look])
                ret.append(rooler.rewards[idx:idx + min_look])
                adv.append(rooler.adv_buf[idx:idx + min_look])
                logp.append(rooler.logprobs[idx:idx + min_look])

        data = dict(obs=obs, act=act, ret=ret,
                    adv=adv, logp=logp)
        return {k: torch.as_tensor(np.array(v), dtype=torch.float32, device=device).reshape(bs,min_look,-1) for k, v in data.items()}


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
    def __init__(self,obs_dim,hidden_size=128):
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
            self.action_var = torch.full((action_dim,), action_std_init).to(device)
        # actor
        self.actor = Actor(state_dim,action_dim,has_continuous_action_space, hidden_space)
        # critic
        self.critic = Critic(state_dim,hidden_space)

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
            # cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = Normal(action_mean, self.action_var)
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
            action_var = self.action_var.expand_as(action_mean)
            # cov_mat = torch.diag_embed(action_var).to(device)
            dist = Normal(action_mean, action_var)
        else:
            action_probs, _ = self.actor(state)
            dist = Categorical(action_probs)
        if self.has_continuous_action_space:
            action_logprobs = dist.log_prob(action)
        else:
            action_logprobs = dist.log_prob(action.squeeze())

        dist_entropy = dist.entropy()
        state_values, _ = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=5e-4, lr_critic=1e-3, gamma=0.99, K_epochs=80, eps_clip=0.3,
                 has_continuous_action_space=True, action_std_init=0.6, hidden_space=128, batch_size=1):

        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        if has_continuous_action_space:
            self.action_std = action_std_init
        self.hidden_space = hidden_space
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.buffer = PPOBuffer()

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

            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)
            # self.buffer.state_values.append(state_val)
            return action.detach().cpu().numpy().flatten(),action_logprob.detach().cpu().numpy(), state_val.detach().cpu().numpy(), h_a, h_c
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device).unsqueeze(0).unsqueeze(0)
                action, action_logprob, state_val, h_a,h_c = self.policy_old.act(state, deter, h_a,h_c)

            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)
            # self.buffer.state_values.append(state_val)

            return action.item(),action_logprob.detach().cpu().numpy(), state_val.detach().cpu().numpy(), h_a, h_c

    def update(self, look_setup=20):
        # Monte Carlo estimate of returns
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            data = self.buffer.sample(look_setup)
            old_states = data['obs']
            old_actions = data['act']
            old_logprobs = data['logp']
            rewards = data['ret']
            advantages = data['adv']
            # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            logprobs,state_values,dist_entropy = [],[],[]
            for _ in range(self.batch_size):
                logprob, state_value, dist_ent = self.policy.evaluate(old_states[_], old_actions[_])
                logprobs.append(logprob)
                state_values.append(state_value)
                dist_entropy.append(dist_ent)
            logprobs, state_values, dist_entropy = torch.stack(logprobs),torch.stack(state_values),torch.stack(dist_entropy)
            logprobs = logprobs.reshape(self.batch_size,-1,1)
            state_values = state_values.reshape(self.batch_size,-1,1)
            dist_entropy = dist_entropy.reshape(self.batch_size,-1,1)
            # match state_values tensor dimensions with rewards tensor
            # state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective RPPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(),max_norm=5)
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(),max_norm=5)

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


