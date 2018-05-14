# -*- coding: UTF-8 -*-
"""
by Howard
using:
- python: 3.6
- Tensorflow: 1.2.1
- gym: 0.9.2
"""
from .DQN_PER_Agent import DQN_PER_Agent as agent
import numpy as np
import math


class PDQN_RePER_Agent(agent):
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
        super(PDQN_RePER_Agent, self).__init__(brain, observation_space_shape, n_actions,  reward_decay, MAX_EPSILON,
                                               MIN_EPSILON,  LAMBDA, memory_size, batch_size, replace_target_iter)
        self.horizon_K = 4
        self.print_q = 0

    def calculating_q_lower_max(self, batch_memory_index):
        q_lower_max = []
        q_next_list = []
        StoredSize = self.memory.getStoredSize()
        for k in range(1, self.horizon_K + 1):
            batch_memory = [self.memory.sumTree.data[index + k][3] if (index + k) < StoredSize
                            else self.memory.sumTree.data[0][3] for index in batch_memory_index]
            q_next = self.brain.predict_target_action(batch_memory)
            q_next_list.append(q_next)

        for offset in range(len(batch_memory_index)):
            index = batch_memory_index[offset]
            q_lower = []
            for k in range(1, min(self.horizon_K, StoredSize - index - 1) + 1):
                cumulative_r = 0
                for i in range(0, k + 1):
                    next_index = index + i
                    cumulative_r += math.pow(self.gamma, i) * self.memory.sumTree.data[next_index][2]
                if self.memory.sumTree.data[index + k][4]:
                    q_next = 0
                else:
                    # q_next = self.brain.predict_target_action([self.memory.sumTree.data[index + k][3]])
                    # print('index:', index, 'q_next', q_next, 'q_next_check', q_next_list[k - 1][offset])
                    q_next = q_next_list[k - 1][offset]

                q_lower.append(cumulative_r + math.pow(self.gamma, k + 1) * np.max(q_next))
            if q_lower:
                q_lower_max.append(max(q_lower))
            else:
                q_lower_max.append(0)
        # print('-------------------------------\n')
        return np.array(q_lower_max)

    def calculating_q_upper_min(self, batch_memory_index):
        q_upper_min = []
        q_before_list = []
        for k in range(1, self.horizon_K + 1):
            batch_memory = [self.memory.sumTree.data[index - k - 1][0] if (index - k - 1) > -1 else self.memory.sumTree.data[0][0] for index in batch_memory_index]
            q_before = self.brain.predict_target_action(batch_memory)
            q_before_list.append(q_before)
        for offset in range(len(batch_memory_index)):
            index = batch_memory_index[offset]
            q_upper = []
            for k in range(1, min(self.horizon_K, index - 1) + 1):
                cumulative_r = 0
                for i in range(0, k + 1):
                    next_index = index - k - 1 + i
                    cumulative_r += math.pow(self.gamma, i - k - 1) * self.memory.sumTree.data[next_index][2]

                # memory_before = self.memory.sumTree.data[index - k - 1]
                # q_before = self.brain.predict_target_action([memory_before[0]])
                # print('index:', index, 'q_before', q_before, 'q_before_check', q_before_list[k - 1][offset])
                q_before = q_before_list[k - 1][offset]
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
            print('\nPDQN_Agent learn_step_counter=', self.learn_step_counter, ' target_params_replaced')
            print('epsilon:', self.epsilon, '\n')
            # print('printq', self.print_q)

        # 从 memory 中随机抽取 batch_size 大小的记忆
        batch_size = min(self.batch_size, self.memory.getStoredSize())
        # ------------------ 这部分和DDQN 不一样 ------------------
        # batch_memory = random.sample(self.memory, batch_size)
        tree_idx, batch_memory, ISWeights = self.memory.sample(batch_size)
        batch_memory_index = tree_idx - self.memory.sumTree.capacity + 1

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

        # self.print_q = q_target
        # ----------------------------------------------------------------------
        # One Hot Encoding
        one_hot_action = np.eye(self.n_actions)[action]
        # 训练 eval 神经网络
        if costFlag:
            abs_errors, cost = self.brain.train(states, q_target, one_hot_action, ISWeights, True)
            self.cost_his.append(cost)
        else:
            abs_errors = self.brain.train(states, q_target, one_hot_action, ISWeights)

        self.memory.batch_update(tree_idx, abs_errors)
        self.memory.batch_nearby_update(tree_idx)
        # brain 中 的 output_graph 需要为 True
        self.brain.output_tensorboard(states, q_target, states_, one_hot_action, ISWeights, self.learn_step_counter)
        self.learn_step_counter += 1


if __name__ == '__main__':
    Brain = brain(n_actions=1, n_features=1, output_graph=True)
    agent = Agent(Brain, n_actions=1, n_features=1,)
    agent.plot_cost()
