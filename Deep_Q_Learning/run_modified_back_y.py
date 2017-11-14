# -*- coding: UTF-8 -*-
import gym
from MLP_Brain import MLP_Brain as brain
from Modified_Agent import Agent
import numpy as np
import math
env = gym.make('CartPole-v0')   # 定义使用 gym 库中的那一个环境
env = env.unwrapped  # 注释掉的话 每局游戏 reward之和最高200
env.seed(1)
print(env.action_space.sample())  # 查看这个环境中可用的 action 有多少个
print(env.observation_space.shape)    # 查看这个环境中可用的 state 的 observation 有多少个
print(env.observation_space.high)   # 查看 observation 最高取值
print(env.observation_space.low)    # 查看 observation 最低取值

# learning_rate 重要
# restore 和 MAX_EPSILON 一起调整
Brain = brain(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    neurons_per_layer=np.array([64]),
    learning_rate=0.00025,
    output_graph=False,
    restore=False,
)
RL = Agent(
    brain=Brain,
    n_actions=env.action_space.n,
    observation_space_shape=env.observation_space.shape,
    reward_decay=0.9,
    replace_target_iter=1000,
    memory_size=100000,
    batch_size=64,
    MAX_EPSILON=0.9,
    LAMBDA=0.0001,
)


def run_game():
    total_steps = 0
    for i_episode in range(1, 1001):

        observation = env.reset()

        totalR = 0
        while True:
            # env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)
            totalR += reward
            RL.store_memory(observation, action, reward, observation_, done)

            if totalR > 100000:
                return 0
            if done:
                if i_episode % 10 == 0:
                    RL.set_y_true()
                    b = RL.batch_size
                    m = len(RL.memory)
                    print('len(RL.memory)', m)
                    if b < m / 2:
                        print('----------------------------------------------------------b<M---------------------------------------')
                        # n = m * m / b
                        # print('n:', n)
                        for i in range(int(m / 2) + b):
                            RL.learn()
                        RL.memory = []
                    else:
                        print('-------------------------------------------------------------------------------------------------')
                        # RL.learn()
                RL.statistical_values(totalR)
                print('episode: ', i_episode,
                      ' epsilon: ', RL.epsilon,
                      'total_reward:', totalR)
                break

            observation = observation_
            total_steps += 1


run_game()
RL.plot_values('back_y version:10 episode\'s memory,if 2 * b < m: RL.learn() m/2 + b times')
