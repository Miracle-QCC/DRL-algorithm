from DDPGModel import *
import gym
import matplotlib.pyplot as plt
import pandas as pd
import logging
import colorlog
from tensorboardX import SummaryWriter
import numpy as np

writer = SummaryWriter('runs/rewards')
logging.basicConfig(filename='logs/reward_log.txt', format='%(asctime)s %(message)s')
colorformat = colorlog.ColoredFormatter("%(log_color)s%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hander = colorlog.StreamHandler()
hander.setFormatter(colorformat)
logger.addHandler(hander)

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Store experimental data
    data = pd.DataFrame(columns=['reward'])

    ddpg = Agent(obs_dim, act_dim)
    trainTimes = 0
    MAX_TotalStep = int(2e6)
    MAX_EPISODE = int(300)
    MAX_STEP = 500
    MaxTrain = 50

    batch_size = 64
    rewardList = []

    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            if episode > 10:
                a = ddpg.get_action(o,deter=(False if episode < 200 else True))
            else:
                a = env.action_space.sample()
            o2, r, d,_ = env.step(a)
            ddpg.store(o, a, r, o2, d)

            o = o2
            ep_reward += r
            if d:
                break
        if episode >= 10:  #Training 50 times per round
            for i in range(MaxTrain):
                trainTimes += 1
                ddpg.update(batch_size)

        data.loc[trainTimes, "reward"] = ep_reward
        writer.add_scalar('ddpg', ep_reward, global_step=trainTimes)
        logger.info('trainTimes: %d - Episode: %d - Reward:%f' % (trainTimes, episode, ep_reward))
        rewardList.append(ep_reward)
        if trainTimes >= MAX_TotalStep:
            break

    data.to_csv("ddpg.csv")
    torch.save(ddpg.ac.pi.state_dict(), 'actor.pt')
    plt.figure()
    plt.plot(np.arange(len(rewardList)), rewardList)
    plt.show()