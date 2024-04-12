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
    env = gym.make('Pendulum-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = (env.action_space.high - env.action_space.low) / 2.0

    # Store experimental data
    data = pd.DataFrame(columns=['reward'])
    batch_size = 8
    ddpg = Agent(obs_dim, act_dim, act_bound,batch_size = batch_size)
    trainTimes = 0
    MAX_TotalStep = int(2e6)
    MAX_EPISODE = int(4e6)
    MAX_STEP = 500
    MaxTrain = 50

    rewardList = []

    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        episode_record = EpisodeBuffer()

        h_a,h_c = ddpg.init_hidden_state(batch_size=1,training=False)
        for j in range(MAX_STEP):
            if episode > 5:
                # print(o.shape)
                a,h_a = ddpg.get_action(o,h_a)
            else:
                a = env.action_space.sample()
            o2, r, d, _ = env.step(a.reshape(-1,))
            episode_record.put([o, a.squeeze(), r, o2, d])
            o = o2
            ep_reward += r
            if d:
                break
        ddpg.store(episode_record)
        if episode >= 10:  #Training 50 times per round
            for i in range(MaxTrain):
                trainTimes += 1
                ddpg.update(batch_size)

        data.loc[trainTimes, "reward"] = ep_reward
        writer.add_scalar('Pendulum-v1_ddpg', ep_reward, global_step=trainTimes)
        logger.info('trainTimes: %d - Episode: %d - Reward:%f' % (trainTimes, episode, ep_reward))
        rewardList.append(ep_reward)
        if trainTimes >= MAX_TotalStep:
            break

    data.to_csv("Pendulum-v1_ddpg.csv")
    torch.save(ddpg.ac.pi.state_dict(), 'actor.pt')
    plt.figure()
    plt.plot(np.arange(len(rewardList)), rewardList)
    plt.show()
