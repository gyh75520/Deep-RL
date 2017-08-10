import gym
from MLP import Neural_Networks as brain
from Agent import Agent
import numpy as np

env = gym.make('CartPole-v0')   # 定义使用 gym 库中的那一个环境
env = env.unwrapped  # 不做这个会有很多限制

print(env.action_space.sample())  # 查看这个环境中可用的 action 有多少个
print(env.observation_space.shape)    # 查看这个环境中可用的 state 的 observation 有多少个
print(env.observation_space.high)   # 查看 observation 最高取值
print(env.observation_space.low)    # 查看 observation 最低取值

# learning_rate 重要
Brain = brain(n_actions=env.action_space.n, n_features=env.observation_space.shape[0], neurons_per_layer=np.array([64]), learning_rate=0.00025, output_graph=True)
RL = Agent(
    brain=Brain,
    n_actions=env.action_space.n,
    observation_space=env.observation_space,
    reward_decay=0.9,
    replace_target_iter=100,
    memory_size=2000,
)

total_steps = 0


for i_episode in range(200):

    observation = env.reset()
    ep_r = 0
    totalR = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, env_reward, done, info = env.step(action)

        # the smaller theta and closer to center the better

        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        totalR += env_reward
        RL.store_memory(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(RL.epsilon, 2),
                  'total_reward:', totalR)
            break

        observation = observation_
        total_steps += 1
RL.plot_cost()
