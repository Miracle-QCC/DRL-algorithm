import math
import matplotlib.pyplot as plt

import torch
from torch import nn
import random
import numpy as np
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
device = 'cuda' if torch.cuda.is_available() else "cpu"


class DQN(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DQN, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,act_dim)
        )

    def forward(self,obs):
        q = self.q(obs)
        return q


class DDQN_Agent:
    # def __init__(self, obs_dim, act_dim, buff_size=int(1e5), gamma=0.99, eps = 0.9, lr=1e-3, tau=0.02):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        ## 创建Q网络和target网络
        self.q = DQN(self.state_space_dim, self.action_space_dim).to(device)
        self.q_targ = DQN(self.state_space_dim, self.action_space_dim).to(device)
        self.q_targ.load_state_dict(self.q.state_dict())
        for parm in self.q_targ.parameters():
            parm.requires_grad = False
        self.buffer = []
        self.criterion = nn.MSELoss()
        self.optim = torch.optim.Adam(self.q.parameters(), lr=self.lr, weight_decay=1e-4)
        self.tau = 0.005
        self.steps = 0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def act(self, obs):

        ### 随机选一个动作
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * self.steps / self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.action_space_dim)
        else:
            s0 = torch.tensor(obs, dtype=torch.float).view(1, -1).to(device)
            a0 = torch.argmax(self.q(s0)).item()
        return a0

    def update(self, curious_net=None):

        ### 数据太少了，不训练
        if len(self.buffer) < self.batch_size:
            return
        samples = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*samples)

        # state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.from_numpy(np.array(state)).to(device).float()
        action = torch.tensor(action).to(device).view(self.batch_size, -1).long()
        reward = torch.tensor(reward).to(device).view(self.batch_size, -1).float()
        next_state = torch.from_numpy(np.array(next_state)).to(device).float()
        done = torch.tensor(done).to(device).float()

        with torch.no_grad():
            if curious_net:
                curious_value = curious_net.value(next_state)
                q_targ = reward + (
                            ((1 - done) * self.gamma) * (torch.max(self.q_targ(next_state).detach(), dim=1)[0] + curious_value)).reshape(-1, 1)

            else:
                q_targ = reward + (((1-done) * self.gamma) * torch.max(self.q_targ(next_state).detach(), dim=1)[0]).reshape(-1, 1)

        q = self.q(state)
        q_cur = q.gather(1, action)

        ### 进行更新
        loss = self.criterion(q_cur, q_targ)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        ### 软更新taget
        for p,p_targ in zip(self.q.parameters(), self.q_targ.parameters()):
            p_targ.data = (1 - self.tau) * p_targ.data + self.tau * p.data




class CloudManufacturingEnvironment:
    def __init__(self, num_tasks=30, num_resources_per_task=5):
        self.num_tasks = num_tasks  # 子任务数量
        self.num_resources_per_task = num_resources_per_task  # 每个子任务的资源数量
        self.current_task = 0  # 当前子任务索引
        self.qos_values = np.random.rand(num_tasks, num_resources_per_task)  # 随机生成的QoS值
        self.qos_values = (self.qos_values - np.mean(self.qos_values,axis=1)) / (np.std(self.qos_values,axis=1) + 1e-4)
        self.state = np.zeros(num_tasks)  # 初始化状态，表示每个子任务的资源匹配状态

    def step(self, action):
        reward = 0
        done = False

        if self.state[self.current_task] == 0:
            self.state[self.current_task] = 1  # 标记当前子任务的资源已匹配
            reward = self.qos_values[self.current_task, action]  # 奖励为该资源的QoS值
            self.current_task += 1  # 转到下一个子任务

        if self.current_task >= self.num_tasks:
            done = True  # 所有子任务都匹配完毕

        return self.state, reward, done

    def reset(self):
        self.state = np.zeros(self.num_tasks)  # 重置状态
        self.current_task = 0  # 重置当前子任务索引
        return self.state

def run():
    num_tasks = 30
    num_resources_per_task = 30
    input_size = num_tasks
    output_size = num_resources_per_task
    params = {
        'gamma': 0.9,
        'epsi_high': 1.0,
        'epsi_low': 0.001,
        'decay': 1000,
        'lr': 0.001,
        'capacity': 1000000,
        'batch_size': 128,
        'state_space_dim': input_size,
        'action_space_dim': output_size,
    }

    agent = DDQN_Agent(**params)
    env = CloudManufacturingEnvironment(num_tasks=num_tasks, num_resources_per_task=num_resources_per_task)
    totalreward = []

    num_episodes = 1000
    batch_size = 64
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            # print("action:",action)
            next_state, reward, done = env.step(action)
            # print("next_state:",next_state)
            agent.put(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.update()

        rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        totalreward.append(total_reward)

    plt.plot(totalreward)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

if __name__ == "__main__":
    run()







