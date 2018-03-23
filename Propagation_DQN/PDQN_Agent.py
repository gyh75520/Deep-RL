# -*- coding: UTF-8 -*-
"""
by Howard
using:
- python: 3.6
- Tensorflow: 1.2.1
- gym: 0.9.2
"""

import numpy as np
import random
import sys
# sys.path.append("..")
from DQN.Agent import Agent as agent
import operator
import math


class Agent(agent):
    def __init__(
        self,
        brain,  # 使用的神经网络
        observation_space_shape,
        n_actions,  # 动作数
        reward_decay=0.9,  # gamma参数
        MAX_EPSILON=0.9,  # epsilon 的最大值
        MIN_EPSILON=0.01,  # epsilon 的最小值
        LAMBDA=0.001,  # speed of decay
        memory_size=500,  # 记忆的大小
        batch_size=32,  # 每次更新时从 memory 里面取多少记忆出来
        replace_target_iter=300,  # 更换 target_net 的步数
    ):
        super(Agent, self).__init__(brain, observation_space_shape, n_actions,  reward_decay, MAX_EPSILON,
                                    MIN_EPSILON,  LAMBDA, memory_size, batch_size, replace_target_iter)
        self.horizon_K = 4
        self.print_q = 0

    def calculating_q_lower_max(self, batch_memory_index):
        q_lower_max = []
        for index in batch_memory_index:
            q_lower = []
            for k in range(1, min(self.horizon_K, len(self.memory) - index - 1) + 1):
                cumulative_r = 0
                for i in range(0, k + 1):
                    next_index = index + i
                    cumulative_r += math.pow(self.gamma, i) * self.memory[next_index][2]
                if self.memory[index + k][4]:
                    q_next = 0
                else:
                    q_next = self.brain.predict_target_action([self.memory[index + k][3]])
                    # if operator.eq(self.memory[index + k + 1][0], self.memory[index + k][3]) is not True:
                    #     print('----------------------------------------------------------')
                    #     print(self.memory[index + k + 1][0], self.memory[index + k][3])
                # print('q_next', q_next)
                q_lower.append(cumulative_r + math.pow(self.gamma, k + 1) * np.max(q_next))
            if q_lower:
                q_lower_max.append(max(q_lower))
            else:
                q_lower_max.append(0)
        return np.array(q_lower_max)

    def calculating_q_upper_min(self, batch_memory_index):
        q_upper_min = []
        for index in batch_memory_index:
            q_upper = []
            for k in range(1, min(self.horizon_K, index - 1) + 1):
                cumulative_r = 0
                for i in range(0, k + 1):
                    next_index = index - k - 1 + i
                    cumulative_r += math.pow(self.gamma, i - k - 1) * self.memory[next_index][2]

                memory_before = self.memory[index - k - 1]
                q_before = self.brain.predict_target_action([memory_before[0]])
                # print('q_before', q_before[0])
                # print('q', (math.pow(self.gamma, -k - 1) * q_before[0][self.memory[index - k - 1][1]]))
                # print('cumulative_r', cumulative_r)

                # q_upper.append((math.pow(self.gamma, -k - 1) * q_before[0][memory_before[1]]) - cumulative_r)
                q_upper.append(math.pow(self.gamma, -k - 1) * np.max(q_before[0]) - cumulative_r)
            if q_upper:
                q_upper_min.append(min(q_upper))
            else:
                q_upper_min.append(0)

        return np.array(q_upper_min)

    def learn(self, costFlag=False):
        # 每隔 replace_target_iter 步 替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.brain.replace_target_params()
            print('\nlearn_step_counter=', self.learn_step_counter, ' target_params_replaced')
            print('epsilon:', self.epsilon, '\n')
            print('printq', self.print_q)

        # 从 memory 中随机抽取 batch_size 大小的记忆
        batch_size = min(self.batch_size, len(self.memory))
        # batch_memory = random.sample(self.memory, batch_size)
        batch_memory_index = random.sample(range(len(self.memory)), batch_size)
        batch_memory = [self.memory[index] for index in batch_memory_index]
        # print('\nbatch_memory_index', batch_memory_index)
        # no_state = np.zeros(self.observation_space_shape)
        states = np.array([o[0] for o in batch_memory])
        states_ = np.array([o[3] for o in batch_memory])
        action = np.array([o[1] for o in batch_memory])
        reward = np.array([o[2] for o in batch_memory])

        q_next = self.brain.predict_target_action(states_)
        '''
        q_target_ = []
        for i in range(0, batch_size):
            q_target_.append(reward[i] + self.gamma * np.max(q_next[i]))
        下面的损失了一些精度
        '''
        # q_target = reward + self.gamma * np.max(q_next, axis=1)

        # ---------------------- difference with DQN ---------------------------
        q_target = []
        for i in range(0, batch_size):
            done = batch_memory[i][4]
            if done:
                q_target.append(reward[i])
            else:
                q_target.append(reward[i] + self.gamma * np.max(q_next[i]))
        q_lower_max = self.calculating_q_lower_max(batch_memory_index)
        q_upper_min = self.calculating_q_upper_min(batch_memory_index)

        for i in range(0, batch_size):
            if q_lower_max[i] < q_upper_min[i]:
                if q_target[i] < q_lower_max[i]:
                    q_target[i] = q_lower_max[i]
                elif q_target[i] > q_upper_min[i]:
                    q_target[i] = q_upper_min[i]

        self.print_q = q_target
        # ----------------------------------------------------------------------
        # One Hot Encoding
        one_hot_action = np.eye(self.n_actions)[action]
        # 训练 eval 神经网络
        if costFlag:
            cost = self.brain.train(states, q_target, one_hot_action, True)
            self.cost_his.append(cost)
        else:
            self.brain.train(states, q_target, one_hot_action)
        # brain 中 的 output_graph 需要为 True
        self.brain.output_tensorboard(states, q_target, states_, one_hot_action, self.learn_step_counter)
        self.learn_step_counter += 1


if __name__ == '__main__':
    Brain = brain(n_actions=1, n_features=1, output_graph=True)
    agent = Agent(Brain, n_actions=1, n_features=1,)
    agent.plot_cost()
