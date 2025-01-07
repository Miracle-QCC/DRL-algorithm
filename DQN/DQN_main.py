import sys

import gymnasium as gym
import math
import matplotlib.pyplot as plt
import torch
import sys
import os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取父目录
parent_dir = os.path.dirname(current_dir)
# 将 models 目录添加到 sys.path
sys.path.append(os.path.join(parent_dir, 'models'))
from DQN import DQN_Agent
from OU_noise import *
import numpy as np

import pandas as pd
import logging
import colorlog
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs/rewards/')
logging.basicConfig(format='%(asctime)s %(message)s')
colorformat = colorlog.ColoredFormatter("%(log_color)s%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

env = gym.make('MountainCar-v0')
env.reset()

if __name__ == '__main__':
    # 噪声类型
    # noise_type = sys.argv[1]  # 'simple_noise' 'hard_noise'
    # # 噪声强度
    # degree = sys.argv[2]  # 0 1 2
    params = {
        'gamma': 0.99,
        'epsi_high': 1.0,
        'epsi_low': 0.05,
        'decay': 100000,
        'lr': 0.001,
        'capacity': 1000000,
        'batch_size': 512,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n,
    }
    for noise_type in ['no', 'simple', 'hard']:
        for degree in [0.01, 0.05]:
            data = pd.DataFrame(columns=['reward'])

            suffix = f'_{noise_type}_{degree}' if noise_type != 'no' else ''
            ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(env.observation_space.shape[0]))
            agent = DQN_Agent(**params)
            totalreward = []

            num_episodes = 1000
            rewards = []
            trainTimes = 0
            epsi = 1.0
            for episode in range(num_episodes):
                state,_ = env.reset()
                total_reward = 0
                done = False
                steps = 0
                while not done and steps < 500:
                    if noise_type == 'no':
                        state = state
                    elif noise_type == 'simple':
                        state = state + np.random.uniform(-0.5, 0.5, state.shape) * degree
                    elif noise_type == 'hard':
                        state = state + ou_noise() * degree
                    action = agent.act(state,epsi)
                    steps += 1
                    # print("action:",action)
                    next_state, reward, done, _, _ = env.step(action)
                    t_reward = 2000 * ((np.sin(3 * next_state[0]) * 0.0025 + 0.5 * next_state[1] * next_state[1]) -
                                      (np.sin(3 * state[0]) * 0.0025 + 0.5 * state[1] * state[1]))
                    if next_state[0] >= 0.5:
                        t_reward += 1
                    # print("next_state:",next_state)
                    agent.put(state, action, t_reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    if done:
                        break
                epsi *= 0.995
                epsi = max(epsi,0.01)
                for _ in range(50):
                    agent.update()
                trainTimes += 40
                rewards.append(total_reward)
                writer.add_scalar(f'MountainCar-v0_DQN{suffix}', total_reward, global_step=episode)
                # print(f"Episode {episode + 1}, Total Reward: {total_reward}")
                logger.info('trainTimes: %d - Episode: %d - Reward:%f  epsi:%f' % (trainTimes, episode, total_reward,epsi))
                totalreward.append(total_reward)
                data.loc[episode, "reward"] = total_reward
            data.to_csv(f"MountainCar-v0_DQN{suffix}.csv")
            torch.save(agent.q.state_dict(),f'ckpt/MountainCar-v0_DQN{suffix}.pt')
