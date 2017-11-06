# -*- coding: UTF-8 -*-
import gym
from MLP_Brain import MLP_Brain as brain
from Agent import Agent
import numpy as np
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
    output_graph=True,
    restore=False,
)
RL = Agent(
    brain=Brain,
    n_actions=env.action_space.n,
    observation_space_shape=env.observation_space.shape,
    reward_decay=0.95,
    replace_target_iter=1000,
    memory_size=100000,
    batch_size=64,
    MAX_EPSILON=0.9,
    LAMBDA=0.001,
)


def set_memory_with_random():
    while len(RL.memory) < RL.memory_size:
        observation = env.reset()
        while True:
            # env.render()
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            RL.store_memory(observation, action, reward, observation_, done)
            if done:
                break
            observation = observation_
    print('set_memory_with_random successful, memory_size:', len(RL.memory))


def run_game(episode, plt_q=False):
    set_memory_with_random()
    total_steps = 0
    # q_change = [[0.03073904, 0.00145001, -0.03088818, -0.03131252]]
    for i_episode in range(episode):
        observation = env.reset()
        if plt_q:
            q_change = [observation]
            action_change = RL.choose_action(observation)
        totalR = 0
        while True:
            # env.render()
            action = RL.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            totalR += reward
            RL.store_memory(observation, action, reward, observation_, done)
            RL.learn()
            if done:
                print('episode: ', i_episode, ' epsilon: ', RL.epsilon, 'total_reward:', totalR)
                if plt_q:
                    RL.statistical_values(totalR, q_change, action_change)
                else:
                    RL.statistical_values(totalR)
                break
            observation = observation_
            total_steps += 1

    # Brain.save()  # 存储神经网络
    RL.plot_values()


run_game(100, True)
