from PPOModel import *
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
    env = gym.make('Pendulum-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = (env.action_space.high - env.action_space.low) / 2.0

    #Store experimental data
    data = pd.DataFrame(columns=['reward'])
    has_continuous_action_space = True
    action_std_init = 1.0
    ppo = PPO(obs_dim, act_dim,has_continuous_action_space=has_continuous_action_space,action_std_init=action_std_init)
    trainTimes = 0
    MAX_EPISODE = int(1000)
    MAX_STEP = 1000
    K_epochs = 80
    rewardList = []
    time_step = 0
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    update_timestep = 4000
    save_model_freq = update_timestep * 10
    # training loop
    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            time_step += 1
            a = ppo.get_action(o)
            o2, r, d, _ = env.step(a)
            ppo.buffer.rewards.append(r)
            ppo.buffer.is_terminals.append(d)
            # ppo.buffer.store(o,a,r,state_val,action_logprob)
            o = o2
            ep_reward += r
            # update RPPO agent
            if time_step % update_timestep == 0:
                ppo.update()
                trainTimes += K_epochs
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo.decay_action_std(action_std_decay_rate, min_action_std)
            if d:
                break

        data.loc[trainTimes,"reward"] = ep_reward
        writer.add_scalar('Pendulum-v1_ppo_my',ep_reward,global_step=trainTimes)
        logger.info('trainTimes: %d - Episode: %d - Reward:%f'%(trainTimes,episode,ep_reward))
        rewardList.append(ep_reward)


    data.to_csv("Pendulum-v1_ppo_my.csv")
    ppo.save('actor.pt')
    plt.figure()
    plt.plot(np.arange(len(rewardList)),rewardList)
    plt.show()
