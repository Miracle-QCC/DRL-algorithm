import torch
import gym
import core as core
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SAC:
    def __init__(self,obs_dim, act_dim,act_bound, actor_critic=core.MLPActorCritic):

        self.act_bound = act_bound
        self.ac = actor_critic(obs_dim, act_dim, act_limit=2.0).to(device)
        self.ac.pi.load_state_dict(torch.load('actor.pht'))
    def get_action(self, o, deterministic=False):
        o = torch.FloatTensor(o).to(device)
        return self.ac.act(o,deterministic)

if __name__ == '__main__':
    env = gym.make('BipedalWalkerHardcore-v3')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = [-env.action_space.high[0], env.action_space.high[0]]
    sac = SAC(obs_dim, act_dim, act_bound)
    t_step = 0
    MAX_EPISODE = 200000
    MAX_STEP = 400


    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            a = sac.get_action(o)
            o2, r, d, _ = env.step(a)
            env.render()

            o = o2
            ep_reward += r
            t_step += 1
            if d:
                break
        print('Episode:', episode, 'Reward:%i' % int(ep_reward))