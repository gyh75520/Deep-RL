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


class Agent:
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
        self.brain = brain
        self.observation_space_shape = observation_space_shape
        self.n_actions = n_actions
        self.gamma = reward_decay
        self.MAX_EPSILON = MAX_EPSILON
        self.MIN_EPSILON = MIN_EPSILON
        self.LAMBDA = LAMBDA
        self.epsilon = MAX_EPSILON
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.cost_his = []

        # 记录学习次数 (用于判断是否更换 target_net 参数)
        self.learn_step_counter = 0
        self.choose_step_counter = 0
        self.reset_epsilon_step = 0
        # 初始化全 0 记忆 [s, a, r, s_]
        self.memory = []
        # self.memory = deque()

    def store_memory(self, s, a, r, s_, done):

        self.memory.append((s, a, r, s_, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def choose_action(self, observation):
        # 增加一个维度[observation]
        observation = observation[np.newaxis, :]
        # epsilon greedy 探索
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            # 选择 q 值最大的 action
            actions_value = self.brain.predict_eval_action(observation)
            action = np.argmax(actions_value)
        self.choose_step_counter += 1
        self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * np.exp(-self.LAMBDA * (self.choose_step_counter - self.reset_epsilon_step))
        return action

    def learn(self):
        # 每隔 replace_target_iter 步 替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.brain.replace_target_params()
            print('\nlearn_step_counter=', self.learn_step_counter, ' target_params_replaced')
            print('epsilon:', self.epsilon, '\n')

        # 从 memory 中随机抽取 batch_size 大小的记忆
        batch_size = min(self.batch_size, len(self.memory))
        batch_memory = random.sample(self.memory, batch_size)

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
        q_target = []
        for i in range(0, batch_size):
            done = batch_memory[i][4]
            if done:
                q_target.append(reward[i])
            else:
                q_target.append(reward[i] + self.gamma * np.max(q_next[i]))

        # One Hot Encoding
        one_hot_action = np.eye(self.n_actions)[action]
        # 训练 eval 神经网络
        self.brain.train(states, q_target, one_hot_action, self.learn_step_counter)

        # brain 中 的 output_graph 需要为 True
        self.brain.output_tensorboard(states, q_target, states_, one_hot_action, self.learn_step_counter)

        # 逐渐减少 epsilon, 降低行为的随机性
        self.learn_step_counter += 1
        # self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * np.exp(-self.LAMBDA * (self.learn_step_counter - self.reset_epsilon_step))

    def reset_epsilon(self):
        self.reset_epsilon_step = self.choose_step_counter + 1

    def statistical_values(self, reward, states=None, action=None):
        if not hasattr(self, 'rewards'):
            self.rewards = []
        self.rewards.append(reward)

        # 是否要画 q 值图
        if states is not None:
            if not hasattr(self, 'q_change_list'):
                self.q_change_list = []
            q_value = self.brain.predict_eval_action(states)
            print('q_change', q_value)
            self.q_change_list.append(q_value[0][action])
            return self.rewards, self.q_change_list

        return self.rewards

    def plot_values(self):
        import matplotlib.pyplot as plt
        # from matplotlib.ticker import MultipleLocator
        if not hasattr(self, 'q_change_list'):
            plt.plot(np.arange(len(self.rewards)), self.rewards)
            plt.ylabel('reward')
            plt.xlabel('episode')
            plt.show()
        else:
            # 出对比图
            # ax = plt.subplot(111)  # 注意:一般都在ax中设置,不再plot中设置
            # 设置主刻度标签的位置,标签文本的格式
            # xmajorLocator = MultipleLocator(20)  # 将x主刻度标签设置为20的倍数
            # ymajorLocator = MultipleLocator(10)  # 将y轴主刻度标签设置为10的倍数
            # ax.xaxis.set_major_locator(xmajorLocator)
            # ax.yaxis.set_major_locator(ymajorLocator)
            plt.plot(np.arange(len(self.rewards)), self.rewards, c='r', label='Rewards')
            plt.plot(np.arange(len(self.q_change_list)), self.q_change_list, c='b', label='Q eval')
            plt.legend(loc='best')
            plt.xlabel('episode')
            plt.grid()
            plt.show()
            # plt.subplot(121)
            # plt.plot(np.arange(len(self.rewards)), self.rewards, c='r', label='Rewards')
            # plt.subplot(122)
            # plt.plot(np.arange(len(self.q_change_list)), self.q_change_list, c='b', label='Q eval')
            # plt.show()


if __name__ == '__main__':
    Brain = brain(n_actions=1, n_features=1, output_graph=True)
    agent = Agent(Brain, n_actions=1, n_features=1,)
    agent.plot_cost()
