# -*- coding: UTF-8 -*-
import gym
from MLP_Brain import MLP_Brain as brain
from Agent import Agent as DDQN_Agent
from Deep_Q_Learning.Agent import Agent as DQN_Agent
import numpy as np
import tensorflow as tf

env = gym.make('Pendulum-v0')   # 定义使用 gym 库中的那一个环境
env.seed(1)
ACTION_SPACE = 11    # 将原本的连续动作分离成 11 个动作


def init_DQN():
    Brain = brain(
        n_actions=ACTION_SPACE,
        n_features=env.observation_space.shape[0],
        neurons_per_layer=np.array([64]),
        learning_rate=0.00025,
        output_graph=False,
        restore=False,
    )
    DQN_agent = DQN_Agent(
        brain=Brain,
        n_actions=ACTION_SPACE,
        observation_space_shape=env.observation_space.shape,
        reward_decay=0.9,
        replace_target_iter=200,
        memory_size=100000,
        MAX_EPSILON=0.9,
        batch_size=64,
        LAMBDA=0.0001,
    )
    return DQN_agent


def init_DDQN():
    Brain = brain(
        n_actions=ACTION_SPACE,
        n_features=env.observation_space.shape[0],
        neurons_per_layer=np.array([64]),
        learning_rate=0.00025,
        output_graph=False,
        restore=False,
    )
    DDQN_agent = DDQN_Agent(
        brain=Brain,
        n_actions=ACTION_SPACE,
        observation_space_shape=env.observation_space.shape,
        reward_decay=0.9,
        replace_target_iter=200,
        memory_size=100000,
        MAX_EPSILON=0.9,
        batch_size=64,
        LAMBDA=0.001,
    )
    return DDQN_agent


def run(Agent, episode_Num):

    total_steps = 0
    for i_episode in range(episode_Num):
        observation = env.reset()
        totalR = 0
        while True:
            # env.render()

            action = Agent.choose_action(observation)
            f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)
            observation_, reward, done, info = env.step(np.array([f_action]))
            totalR += reward
            reward /= 10
            Agent.store_memory(observation, action, reward, observation_, done)
            if total_steps > 50:
                Agent.learn()

            if done:
                statistical_reward = Agent.statistical_values(totalR)
                print('episode: ', i_episode,
                      ' epsilon: ', round(Agent.epsilon, 2),
                      'total_reward:', totalR)
                break

            observation = observation_
            total_steps += 1
    return statistical_reward


def compare():
    episode_Num = 40

    DDQN_graph = tf.Graph()
    with DDQN_graph.as_default():
        Agent = init_DDQN()
        DDQN_statistical_reward = run(Agent, episode_Num)
    DQN_graph = tf.Graph()
    with DQN_graph.as_default():
        Agent = init_DQN()
        DQN_statistical_reward = run(Agent, episode_Num)

    # 出对比图
    import matplotlib.pyplot as plt
    plt.plot(np.array(DQN_statistical_reward), c='r', label='DQN')
    plt.plot(np.array(DDQN_statistical_reward), c='b', label='DDQN')
    plt.legend(loc='best')
    plt.ylabel('Reward')
    plt.xlabel('Episode steps')
    plt.grid()
    plt.show()


compare()
