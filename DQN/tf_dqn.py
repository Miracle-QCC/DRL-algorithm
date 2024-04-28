import random

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

def create_model(n,a):
    input = keras.Input(shape=(n))
    x = keras.layers.Dense(64,activation='relu')(input)
    x = keras.layers.Dense(a)(x)

    model = keras.Model(input,x)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model

class DQN_Agent:
    # def __init__(self, obs_dim, act_dim, buff_size=int(1e5), gamma=0.99, eps = 0.9, lr=1e-3, tau=0.02):
    def __init__(self,):
        self.q = create_model(4,2)
        self.q_tart = create_model(4,2)
        self.buffer = []
        self.tau = 0.005
        self.steps = 0
        self.capacity = 10000
        self.eps = 1.0
        self.action_space_dim = 2
        self.batch_size = 64
        self.steps = 0

    def hard_update(self):
        weights_model1 = self.q.get_weights()
        self.q_tart.set_weights(weights_model1)

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def act(self, obs):
        ### 随机选一个动作
        self.steps += 1

        if random.random() < self.eps:
            a0 = random.randrange(self.action_space_dim)
        else:
            q = self.q(obs.reshape(1,-1))
            a0 = np.argmax(q)
        return a0

    def update(self, curious_net=None):
        self.steps += 1
        ### 数据太少了，不训练
        if len(self.buffer) < self.batch_size:
            return
        samples = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*samples)

        # state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = np.array(state)
        action = np.array(action)
        reward = np.array(reward)
        next_state = np.array(next_state)
        done = np.array(done)

        next_q = tf.reduce_max(self.q_tart(next_state), axis=1)
        y = reward + 0.9 * next_q * (1 - done)
        # train plus model
        with tf.GradientTape() as tape:
            # get action and batch index for each game
            index_and_actions = tf.stack([tf.range(self.batch_size), action], axis=1)
            # prediction is plus model evaluating current board
            pred = tf.gather_nd(self.q(state),index_and_actions)
            # compute loss
            loss = keras.losses.mse(y, pred)

        # take a step of SGD
        gradients = tape.gradient(loss, self.q.trainable_variables)
        self.q.optimizer.apply_gradients(zip(gradients, self.q.trainable_variables))

        if self.steps % 50 == 0:
            self.hard_update()
        # ### 软更新taget
        # for p,p_targ in zip(self.q.parameters(), self.q_targ.parameters()):
        #     p_targ.data = (1 - self.tau) * p_targ.data + self.tau * p.data

if __name__ == '__main__':
    agent = DQN_Agent()
    env = gym.make('CartPole-v1')
    for episode in range(200):
        state = env.reset()
        total_reward = 0
        done = False
        agent.eps *= 0.9
        while not done:
            action = agent.act(state)
            # print("action:",action)
            next_state, reward, done,_ = env.step(action)
            # print("next_state:",next_state)
            agent.put(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.update()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
