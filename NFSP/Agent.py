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
        information_state_shape,  # 信息状态的 shape
        n_actions,  # 动作数
        reward_decay=0.9,  # gamma参数
        MAX_EPSILON=0.08,  # epsilon 的最大值
        MIN_EPSILON=0.00,  # epsilon 的最小值
        LAMBDA=0.0001,  # speed of decay
        RL_memory_size=6000,  # RL记忆的大小 600k
        SL_memory_size=200000,  # SL记忆的大小 20m
        RL_batch_size=256,  # 每次更新时从 RL_memory 里面取多少记忆出来
        SL_batch_size=256,  # 每次更新时从 SL_memory 里面取多少记忆出来
        replace_target_iter=1000,  # 更换 target_net 的步数 1000
        eta=0.1,  # anticipatory parameter
    ):
        self.brain = brain
        self.information_state_shape = information_state_shape
        self.n_actions = n_actions
        self.gamma = reward_decay
        self.MAX_EPSILON = MAX_EPSILON
        self.MIN_EPSILON = MIN_EPSILON
        self.LAMBDA = LAMBDA
        self.epsilon = MAX_EPSILON
        self.replace_target_iter = replace_target_iter
        self.RL_memory_size = RL_memory_size
        self.SL_memory_size = SL_memory_size
        self.RL_batch_size = RL_batch_size
        self.SL_batch_size = SL_batch_size
        self.policy_sigma = 0  # 策略sigma
        self.eta = eta

        self.ap_net_cost = []
        self.eval_net_cost = []

        # 记录学习次数 (用于判断是否更换 target_net 参数)
        self.learn_step_counter = 0
        # 记录reservoir sampling 的次数
        self.reservoir_step = 0

        # 初始化全 0 记忆 [s, a, r, s_]
        self.RL_memory = []
        # 初始化全 0 记忆 [s, a]
        self.SL_memory = []

    def set_policy_sigma(self):
        # 设置 策略sigma
        if np.random.uniform() < self.eta:
            self.policy_sigma = 1
        else:
            self.policy_sigma = 2
        print('\npolicy_sigma', self.policy_sigma)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]  # 增加一个维度[observation]
        if self.policy_sigma == 0:
            raise Exception('Please set_policy_sigma before choose_action')
        elif self.policy_sigma == 1:
            # epsilon greedy 探索
            if np.random.uniform() < self.epsilon:
                action = np.random.randint(0, self.n_actions)
            else:
                # 选择 q 值最大的 action
                actions_value = self.brain.predict_eval_action(observation)
                print('\naction', actions_value)
                action = np.argmax(actions_value)
        elif self.policy_sigma == 2:
            actions_probability = self.brain.predict_ap_action_probability(observation)
            print('actions_probability', actions_probability)
            random_num = np.random.uniform()
            action = 0
            for a in actions_probability[0]:
                if random_num >= a:
                    random_num -= a
                    action += 1
                else:
                    break
        return action

    def store_memory(self, s, a, r, s_, done):

        self.RL_memory.append((s, a, r, s_, done))
        if len(self.RL_memory) > self.RL_memory_size:
            self.RL_memory.pop(0)

        if self.policy_sigma == 1:
            self._reservoir_sampling((s, a))

    # exponentially-averaged reservoir sampling
    def _reservoir_sampling(self, sample):
        if self.reservoir_step < self.SL_memory_size:
            self.SL_memory.append(sample)
        else:
            index = np.random.randint(0, self.SL_memory_size)
            if np.random.uniform() < 0.25:
                self.SL_memory[index] = sample
        self.reservoir_step += 1

    def learn(self):
        # 每隔 replace_target_iter 步 替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.brain.replace_target_params()
            print('\nlearn_step_counter=', self.learn_step_counter, ' target_params_replaced')
            print('epsilon:', self.epsilon, '\n')

        # ------------------ 训练 eval 神经网络 ------------------
        # 从 memory 中随机抽取 RL_batch_size 大小的记忆
        RL_batch_size = min(self.RL_batch_size, len(self.RL_memory))
        RL_batch_memory = random.sample(self.RL_memory, RL_batch_size)

        no_state = np.zeros(self.information_state_shape)
        eval_states = np.array([o[0] for o in RL_batch_memory])
        eval_states_ = np.array([(no_state if o[3] is None else o[3]) for o in RL_batch_memory])
        eval_action = np.array([o[1] for o in RL_batch_memory])
        reward = np.array([o[2] for o in RL_batch_memory])

        q_next = self.brain.predict_target_action(eval_states_)

        q_target = []
        for i in range(0, RL_batch_size):
            done = RL_batch_memory[i][4]
            if done:
                q_target.append(reward[i])
            else:
                q_target.append(reward[i] + self.gamma * np.max(q_next[i]))

        # One Hot Encoding
        one_hot_eval_action = np.eye(self.n_actions)[eval_action]
        # 训练 eval 神经网络
        self.brain.train_eval_net(eval_states, q_target, one_hot_eval_action, self.learn_step_counter)

        # ------------------ 训练 average_policy 神经网络 ------------------
        # 从 memory 中随机抽取 RL_batch_size 大小的记忆
        SL_batch_size = min(self.SL_batch_size, len(self.SL_memory))
        if SL_batch_size > 0:
            SL_batch_memory = random.sample(self.SL_memory, SL_batch_size)
            ap_states = np.array([o[0] for o in SL_batch_memory])
            ap_action = np.array([o[1] for o in SL_batch_memory])

            # One Hot Encoding
            one_hot_ap_action = np.eye(self.n_actions)[ap_action]
            # 训练 ap 神经网络
            self.brain.train_ap_net(ap_states,  one_hot_ap_action, self.learn_step_counter)
            # brain 中 的 output_graph 需要为 True
            self.brain.output_tensorboard(ap_states, one_hot_ap_action, one_hot_eval_action, eval_states, q_target, eval_states_, self.learn_step_counter)

        # 逐渐减少 epsilon, 降低行为的随机性
        self.learn_step_counter += 1
        self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * np.exp(-self.LAMBDA * self.learn_step_counter)

    def plot_cost(self):
        # cost 曲线
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.eval_net_cost)), self.eval_net_cost)
        plt.ylabel('eval_net_cost')
        plt.xlabel('training steps')

        plt.plot(np.arange(len(self.ap_net_cost)), self.ap_net_cost)
        plt.ylabel('ap_net_cost')
        plt.xlabel('training steps')
        plt.show()


if __name__ == '__main__':
    from Brain import Brain
    Brain = Brain(n_actions=1, n_features=4, output_graph=True)
    agent = Agent(Brain, n_actions=2, information_state_shape=4)
    agent.plot_cost()
