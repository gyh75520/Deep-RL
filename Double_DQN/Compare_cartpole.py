# -*- coding: UTF-8 -*-
import gym
from MLP_Brain import MLP_Brain as brain
from DDQN_Agent import Agent as DDQN_Agent
from DQN.Agent import Agent as DQN_Agent
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')   # 定义使用 gym 库中的那一个环境
env = env.unwrapped  # 注释掉的话 每局游戏 reward之和最高200
env.seed(1)

n_actions = env.action_space.n
n_features = env.observation_space.shape[0]
neurons_per_layer = np.array([16, 32, 32])
learning_rate = 0.00025
output_graph = False
restore = False

reward_decay = 0.95
replace_target_iter = 1000
memory_size = 100000
batch_size = 32
MAX_EPSILON = 0.9
LAMBDA = 0.0001


def init_DQN():
    Brain = brain(
        n_actions=n_actions,
        n_features=n_features,
        neurons_per_layer=neurons_per_layer,
        learning_rate=learning_rate,
        output_graph=output_graph,
        restore=restore,
    )
    DQN_agent = DQN_Agent(
        brain=Brain,
        n_actions=n_actions,
        observation_space_shape=env.observation_space.shape,
        reward_decay=reward_decay,
        replace_target_iter=replace_target_iter,
        memory_size=memory_size,
        MAX_EPSILON=MAX_EPSILON,
        batch_size=batch_size,
        LAMBDA=LAMBDA,
    )
    return DQN_agent


def init_DDQN():
    Brain = brain(
        n_actions=n_actions,
        n_features=n_features,
        neurons_per_layer=neurons_per_layer,
        learning_rate=learning_rate,
        output_graph=output_graph,
        restore=restore,
    )
    DDQN_agent = DDQN_Agent(
        brain=Brain,
        n_actions=n_actions,
        observation_space_shape=env.observation_space.shape,
        reward_decay=0.948,
        replace_target_iter=replace_target_iter,
        memory_size=memory_size,
        MAX_EPSILON=MAX_EPSILON,
        batch_size=batch_size,
        LAMBDA=LAMBDA,
    )
    return DDQN_agent


def run(RL, episode_Num, plt_q=True):

    # set_memory_with_random()
    total_steps = 0
    # q_change = [[0.03073904, 0.00145001, -0.03088818, -0.03131252]]
    for i_episode in range(episode_Num):
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
                    rewards, q_change_eval = RL.statistical_values(totalR, q_change, action_change)
                else:
                    rewards = RL.statistical_values(totalR)
                    q_change_eval = []
                break
            observation = observation_
            total_steps += 1
    return rewards, q_change_eval


def compare():
    rum_times = 1
    episode_Num = 400

    DDQN_rewards = []
    DQN_rewards = []
    DDQN_av_rewards = np.zeros(episode_Num)
    DQN_av_rewards = np.zeros(episode_Num)

    DQN_q_change_evals = []
    DDQN_q_change_evals = []
    DQN_av_q_change_evals = np.zeros(episode_Num)
    DDQN_av_q_change_evals = np.zeros(episode_Num)
    for i in range(rum_times):
        DQN_graph = tf.Graph()
        with DQN_graph.as_default():
            Agent = init_DQN()
            DQN_reward, DQN_q_change_eval = run(Agent, episode_Num)
            DQN_rewards.append(DQN_reward)
            DQN_q_change_evals.append(DQN_q_change_eval)
        DDQN_graph = tf.Graph()
        with DDQN_graph.as_default():
            Agent_DD = init_DDQN()
            DDQN_reward, DDQN_q_change_eval = run(Agent_DD, episode_Num)
            DDQN_rewards.append(DDQN_reward)
            DDQN_q_change_evals.append(DDQN_q_change_eval)

    for i in range(rum_times):
        DDQN_av_rewards += np.array(DDQN_rewards[i])
        DQN_av_rewards += np.array(DQN_rewards[i])
        DQN_av_q_change_evals += np.array(DQN_q_change_evals[i])
        DDQN_av_q_change_evals += np.array(DDQN_q_change_evals[i])
    DDQN_av_rewards = DDQN_av_rewards / rum_times
    DQN_av_rewards = DQN_av_rewards / rum_times
    DQN_av_q_change_evals = DQN_av_q_change_evals / rum_times
    DDQN_av_q_change_evals = DDQN_av_q_change_evals / rum_times
    # 出对比图
    import matplotlib.pyplot as plt
    plt.plot(np.array(DQN_av_rewards), c='r', label='DQN')
    plt.plot(np.array(DDQN_av_rewards), c='b', label='DDQN')
    # plt.plot(np.array(DQN_av_q_change_evals), c='r', label='DQN')
    # plt.plot(np.array(DDQN_av_q_change_evals), c='b', label='DDQN')
    plt.legend(loc='best')
    # plt.ylabel('Q Value')
    plt.ylabel('Reward')
    plt.xlabel('Episode steps')
    plt.grid()
    plt.show()


compare()
