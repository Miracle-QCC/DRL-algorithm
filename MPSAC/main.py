from SACModel import *
import gym
import matplotlib.pyplot as plt
import pandas as pd
import logging
import colorlog
from tensorboardX import SummaryWriter

writer = SummaryWriter('runs/rewards')
logging.basicConfig(filename='logs/reward_log.txt',format='%(asctime)s %(message)s')
colorformat = colorlog.ColoredFormatter("%(log_color)s%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hander = colorlog.StreamHandler()
hander.setFormatter(colorformat)
logger.addHandler(hander)

if __name__ == '__main__':
    from metadrive import SafeMetaDriveEnv

    config = dict(environment_num=200, start_seed=0, use_lateral=True)
    env = SafeMetaDriveEnv(config)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = env.action_space.high[0]

    #Store experimental data
    data = pd.DataFrame(columns=['reward'])

    sac = SAC(obs_dim, act_dim, act_bound)
    t_step = 0
    MAX_EPISODE = 8000
    MAX_T_STEP = int(4e6)
    MAX_STEP = 500
    batch_size = 128
    rewardList = []
    Max_e_train = 100

    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            if episode > 10:
                a = sac.get_action(o)
            else:
                a = env.action_space.sample()
            o2, r, d, info = env.step(a)
            r = r - info["total_cost"]
            sac.store(o, a, r, o2, d)

            if episode >= 10:
                sac.update(batch_size)

            o = o2
            ep_reward += r
            t_step += 1
            if d:
                break
        data.loc[t_step, "reward"] = ep_reward
        writer.add_scalar('MetaDrive_MPSAC', ep_reward, global_step=t_step)
        logger.info('T_step: %d - Episode: %d - Reward:%f' % (t_step, episode, ep_reward))
        rewardList.append(ep_reward)
        if t_step >= MAX_T_STEP:
            break



    data.to_csv("MetaDrive_MPSAC.csv")
    torch.save(sac.ac.pi.state_dict(), 'MPSAC-actor.pht')
    plt.figure()
    plt.plot(np.arange(len(rewardList)),rewardList)
    plt.show()
