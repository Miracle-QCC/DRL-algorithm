import torch
import gym
from DDPGModel import Agent
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = (env.action_space.high - env.action_space.low) / 2.0
    t_step = 0
    MAX_EPISODE = 200000
    MAX_STEP = 400
    ddpg = Agent(obs_dim,act_dim,act_bound)
    ddpg.ac.pi.load_state_dict(torch.load('actor.pt'))
    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        h_a, h_c = ddpg.init_hidden_state(batch_size=1, training=False)
        for j in range(MAX_STEP):

            a, h_a = ddpg.get_action(o, h_a)
            o, r, d, _ = env.step(a.reshape(-1, ))
            ep_reward += r
            if d:
                break
        print(f"{episode} : {ep_reward}")