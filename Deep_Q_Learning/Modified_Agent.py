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
from Agent import Agent as agent
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
        self.store_memory_step = 0

    def store_memory(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def set_y_true(self):
        length = len(self.memory)
        self.y_true = np.zeros(length)
        temp = 0
        for i in range(length - 1, -1, -1):
            self.y_true[i] = self.memory[i][2] + self.gamma * temp
            temp = self.y_true[i]
        # print('\ny_true', self.y_true)

    def learn(self):
        # 每隔 replace_target_iter 步 替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.brain.replace_target_params()
            print('\nlearn_step_counter=', self.learn_step_counter, ' target_params_replaced')
            print('epsilon:', self.epsilon, '\n')

        # 从 memory 中随机抽取 batch_size 大小的记忆
        batch_size = min(self.batch_size, len(self.memory))
        # batch_memory = random.sample(self.memory, batch_size)
        batch_memory_index = random.sample(range(len(self.memory)), batch_size)
        batch_memory = [self.memory[i] for i in batch_memory_index]
        # print('\nbatch_memory_index', batch_memory_index)
        # no_state = np.zeros(self.observation_space_shape)
        states = np.array([o[0] for o in batch_memory])
        states_ = np.array([o[3] for o in batch_memory])
        action = np.array([o[1] for o in batch_memory])
        reward = np.array([o[2] for o in batch_memory])

        # 获取 q_next (target_net 产生的 q) 和 q_eval(eval_net 产生的 q)
        # q_next_is_end = np.zeros(self.n_actions)
        # self.brain.predict_target_action([s]) 输出 [[x,x]] ravel() 去掉外层list [x,x]
        # q_next = np.array([(q_next_is_end if o[4] is True else self.brain.predict_target_action([o[3]]).ravel()) for o in batch_memory])
        q_next = self.brain.predict_target_action(states_)
        '''
        q_target_ = []
        for i in range(0, batch_size):
            q_target_.append(reward[i] + self.gamma * np.max(q_next[i]))
        下面的损失了一些精度
        '''
        # q_target = reward + self.gamma * np.max(q_next, axis=1)

        # -------------------------------different-----------------------------------------
        # batch_y_true = random.sample(self.y_true, batch_size)

        q_target = []
        for i in range(0, batch_size):
            done = batch_memory[i][4]
            if done:
                q_target.append(reward[i])
            else:
                next_index = batch_memory_index[i] + 1
                if next_index < (len(self.memory) - 1):
                    # print('next_index', next_index, self.memory[next_index - 1])
                    next_action = self.memory[next_index][1]
                    next_y_true = self.y_true[next_index]
                    q_next[i][next_action] = next_y_true
                    q_target.append(reward[i] + self.gamma * np.max(q_next[i]))
                else:
                    q_target.append(reward[i] + self.gamma * np.max(q_next[i]))

        # One Hot Encoding
        one_hot_action = np.eye(self.n_actions)[action]
        # 训练 eval 神经网络
        self.brain.train(states, q_target, one_hot_action, self.learn_step_counter)

        # brain 中 的 output_graph 需要为 True
        self.brain.output_tensorboard(states, q_target, states_, one_hot_action, self.learn_step_counter)
        self.learn_step_counter += 1


if __name__ == '__main__':
    Brain = brain(n_actions=1, n_features=1, output_graph=True)
    agent = Agent(Brain, n_actions=1, n_features=1,)
    agent.plot_cost()
