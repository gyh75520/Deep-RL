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
from Brain import Brain


class Agent:
    def __init__(
        self,
        brain,  # 使用的神经网络
        information_state_shape,
        n_actions,  # 动作数
        reward_decay=0.9,  # gamma参数
        MAX_EPSILON=0.9,  # epsilon 的最大值
        MIN_EPSILON=0.01,  # epsilon 的最小值
        LAMBDA=0.001,  # speed of decay
        RL_memory_size=500,  # RL记忆的大小
        SL_memory_size=500,  # SL记忆的大小
        RL_batch_size=32,  # 每次更新时从 RL_memory 里面取多少记忆出来
        SL_batch_size=32,  # 每次更新时从 RL_memory 里面取多少记忆出来
        replace_target_iter=300,  # 更换 target_net 的步数
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
        self.policy_sigma = 0
        self.eta = eta

        self.ap_net_cost = []
        self.eval_net_cost = []

        # 记录学习次数 (用于判断是否更换 target_net 参数)
        self.learn_step_counter = 0

        # 初始化全 0 记忆 [s, a, r, s_]
        self.RL_memory = []
        self.SL_memory = []

    def set_policy_sigma(self):
        if np.random.uniform() < self.eta:
            self.policy_sigma = 1
        else:
            self.policy_sigma = 2
        print('\npolicy_sigma', self.policy_sigma)

    def choose_action(self, observation):
        # 增加一个维度[observation]
        observation = observation[np.newaxis, :]
        if self.policy_sigma == 0:
            print('Please set_policy_sigma before choose_action')
            exit()
        elif self.policy_sigma == 1:
            # epsilon greedy 探索
            if np.random.uniform() < self.epsilon:
                action = np.random.randint(0, self.n_actions)
            else:
                # 选择 q 值最大的 action
                actions_value = self.brain.predict_eval_action(observation)
                action = np.argmax(actions_value)
        elif self.policy_sigma == 2:
            actions_probability = self.brain.predict_ap_action_probability(observation)
            # print('actions_probability', actions_probability)
            random_num = np.random.uniform()
            action = 0
            for a in actions_probability[0]:
                if random_num >= a:
                    random_num -= a
                    action += 1
                else:
                    break
        return action

    def store_memory(self, s, a, r, s_):

        self.RL_memory.append((s, a, r, s_))
        if len(self.RL_memory) > self.RL_memory_size:
            self.RL_memory.pop(0)

        if self.policy_sigma == 1:
            self.SL_memory.append((s, a))
            if len(self.SL_memory) > self.SL_memory_size:
                self.SL_memory.pop(0)

    def learn(self):
        # 每隔 replace_target_iter 步 替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.brain.replace_target_params()
            print('\nlearn_step_counter=', self.learn_step_counter, ' target_params_replaced')
            print('epsilon:', self.epsilon, '\n')

        # ------------------ 训练 average_policy 神经网络 ------------------
        # 从 memory 中随机抽取 RL_batch_size 大小的记忆
        SL_batch_size = min(self.SL_batch_size, len(self.SL_memory))
        if SL_batch_size > 0:
            SL_batch_memory = random.sample(self.SL_memory, SL_batch_size)
            states = np.array([o[0] for o in SL_batch_memory])
            action = np.array([o[1] for o in SL_batch_memory])

            # One Hot Encoding
            one_hot_action = np.eye(self.n_actions)[action]

            cost_ap = self.brain.train_ap_net(states, one_hot_action, self.learn_step_counter)
            self.ap_net_cost.append(cost_ap)

        # ------------------ 训练 eval 神经网络 ------------------
        # 从 memory 中随机抽取 RL_batch_size 大小的记忆
        RL_batch_size = min(self.RL_batch_size, len(self.RL_memory))
        RL_batch_memory = random.sample(self.RL_memory, RL_batch_size)

        no_state = np.zeros(self.information_state_shape)
        states = np.array([o[0] for o in RL_batch_memory])
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in RL_batch_memory])
        action = np.array([o[1] for o in RL_batch_memory])
        reward = np.array([o[2] for o in RL_batch_memory])
        # 获取 q_next (target_net 产生的 q) 和 q_eval(eval_net 产生的 q)
        q_next = self.brain.predict_target_action(states_)
        q_eval = self.brain.predict_eval_action(states)

        # 计算 q_target
        q_target = q_eval.copy()
        # print('\nstates:', states.shape)
        # print('\nq_target:', q_target.shape)
        batch_index = np.arange(RL_batch_size, dtype=np.int32)
        eval_act_index = action.astype(int)
        # action 必须是 0,1,2... print('eval_act_index:', eval_act_index)

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        cost_eval = self.brain.train_eval_net(states, q_target, self.learn_step_counter)
        self.eval_net_cost.append(cost_eval)

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
    Brain = Brain(n_actions=1, n_features=4, output_graph=True)
    agent = Agent(Brain, n_actions=2, information_state_shape=4)
    agent.plot_cost()
