import torch
import gym
from DDPGModel import Agent
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    env = gym.make('BipedalWalkerHardcore-v3')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = (env.action_space.high - env.action_space.low) / 2.0
    t_step = 0
    MAX_EPISODE = 200000
    MAX_STEP = 400
    ddpg = Agent(obs_dim,act_dim,act_bound)
    ddpg.ac.pi.load_state_dict(torch.load('actor.pht'))
    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            a = ddpg.get_action(o)
            o2, r, d, _ = env.step(a)
            env.render()

            o = o2
            ep_reward += r
            t_step += 1
            if d:
                break
        print('Episode:', episode, 'Reward:%i' % int(ep_reward))