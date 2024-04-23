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
    env = gym.make('Pendulum-v0')
    obs_dim = env.observation_space.shape[0]
    try:
        act_dim = env.action_space.shape[0]
        has_continuous_action_space = True
    except:
        has_continuous_action_space = False
        act_dim = env.action_space.n

    #Store experimental data
    data = pd.DataFrame(columns=['reward'])
    action_std_init = 1.0
    K_epochs = 40
    ppo = PPO(obs_dim, act_dim,has_continuous_action_space=has_continuous_action_space,action_std_init=action_std_init,K_epochs=K_epochs)
    trainTimes = 0
    MAX_EPISODE = int(1000)
    MAX_STEP = 200
    batch_size = 4
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
        rooler_buffer = RolloutBuffer()
        for j in range(MAX_STEP):
            time_step += 1
            a,logp,s_v = ppo.get_action(o)
            o2, r, d, _ = env.step(a)
            # ppo.buffer.rewards.append(r)
            # ppo.buffer.is_terminals.append(d)
            rooler_buffer.store(o,a,r/10,s_v,logp)
            # ppo.buffer.store(o,a,r,state_val,action_logprob)
            o = o2
            ep_reward += r
            # update RPPO agent
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo.decay_action_std(action_std_decay_rate, min_action_std)
            if d:
                break
        rooler_buffer.finish_path()
        ppo.buffer.append(rooler_buffer)
        if len(ppo.buffer) == batch_size:
            ppo.update()
            trainTimes += K_epochs
        data.loc[trainTimes,"reward"] = ep_reward
        writer.add_scalar('Pendulum-v1_ppo_my',ep_reward,global_step=trainTimes)
        logger.info('trainTimes: %d - Episode: %d - Reward:%f'%(trainTimes,episode,ep_reward))
        rewardList.append(ep_reward)


    data.to_csv("Pendulum-v1_ppo_my.csv")
    ppo.save('actor.pt')
    plt.figure()
    plt.plot(np.arange(len(rewardList)),rewardList)
    plt.show()
