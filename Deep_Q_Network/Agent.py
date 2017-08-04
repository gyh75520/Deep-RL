"""
by Howard
using:
- python: 3.6
- Tensorflow: 1.2.1
- gym: 0.9.2
"""

import numpy as np
# from RL_Brain import Neural_Networks as brain


class Agent:
    def __init__(
        self,
        brain,  # 使用的神经网络
        n_features,  # 特征数
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
        self.n_features = n_features
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

        # 初始化全 0 记忆 [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

    def store_memory(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # 总 memory 大小是固定的, 如果超出总大小, 旧 memory 就被新 memory 替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # 统一shape (1, size_of_observation)
        observation = observation[np.newaxis, :]
        # epsilon greedy 探索
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            # 选择 q 值最大的 action
            actions_value = self.brain.predict_eval_action(observation)
            action = np.argmax(actions_value)
        return action

    def learn(self):
        # 每隔 replace_target_iter 步 替换 target_net 参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.brain.replace_target_params()
            print('\nlearn_step_counter=', self.learn_step_counter, ' target_params_replaced')
            print('epsilon:', self.epsilon, '\n')

        # 从 memory 中随机抽取 batch_size 大小的记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 获取 q_next (target_net 产生了 q) 和 q_eval(eval_net 产生的 q)
        q_next = self.brain.predict_eval_action(batch_memory[:, -self.n_features:])
        q_eval = self.brain.predict_target_action(batch_memory[:, :self.n_features])

        # 计算 q_target
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # 训练 eval 神经网络
        self.cost = self.brain.train(batch_memory[:, :self.n_features], q_target, self.learn_step_counter)
        self.cost_his.append(self.cost)

        # 逐渐减少 epsilon, 降低行为的随机性
        self.learn_step_counter += 1
        self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) * np.exp(-self.LAMBDA * self.learn_step_counter)

    def plot_cost(self):
        # cost 曲线
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


if __name__ == '__main__':
    Brain = brain(n_actions=1, n_features=1, output_graph=True)
    agent = Agent(Brain, n_actions=1, n_features=1,)
    agent.plot_cost()
