"""
by Howard
using:
- python: 3.6
- Tensorflow: 1.2.1
- gym: 0.9.2
"""

import numpy as np
import random
# from RL_Brain import Neural_Networks as brain


class Agent:
    def __init__(
        self,
        brain,  # 使用的神经网络
        observation_space,
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
        self.observation_space = observation_space
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
        self.memory = []

    def store_memory(self, s, a, r, s_):

        self.memory.append((s, a, r, s_))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

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
        batch_size = min(self.batch_size, len(self.memory))

        batch_memory = random.sample(self.memory, batch_size)

        no_state = np.zeros(self.observation_space.shape)
        states = np.array([o[0] for o in batch_memory])
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch_memory])
        action = np.array([o[1] for o in batch_memory])
        reward = np.array([o[2] for o in batch_memory])
        # 获取 q_next (target_net 产生的 q) 和 q_eval(eval_net 产生的 q)
        q_next = self.brain.predict_target_action(states_)
        q_eval = self.brain.predict_eval_action(states)

        # 计算 q_target
        q_target = q_eval.copy()
        # print('\nstates:', states.shape)
        # print('\nq_target:', q_target.shape)
        batch_index = np.arange(batch_size, dtype=np.int32)
        eval_act_index = action.astype(int)
        # action 必须是 0,1,2... print('eval_act_index:', eval_act_index)

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # 训练 eval 神经网络
        self.cost = self.brain.train(states, q_target, self.learn_step_counter)
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
