import gym
from CNN import CNN as brain
from Agent import Agent
import numpy as np

env = gym.make('Breakout-v0')   # 定义使用 gym 库中的那一个环境
env = env.unwrapped  # 不做这个会有很多限制

print(env.action_space.n)  # 查看这个环境中可用的 action 有多少个
print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
# print(env.observation_space.high)   # 查看 observation 最高取值
# print(env.observation_space.low)    # 查看 observation 最低取值


Brain = brain(n_actions=env.action_space.n, n_features_width=210, n_features_height=160, n_features_depth=3, neurons_per_layer=np.array([32, 64]), output_graph=True)
RL = Agent(
    brain=Brain,
    n_actions=env.action_space.n,
    observation_space=env.observation_space,
    reward_decay=0.9,
    replace_target_iter=100,
    memory_size=2000,
    batch_size=64,
)

total_steps = 0


for i_episode in range(20):

    observation = env.reset()
    ep_r = 0
    totalR = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, env_reward, done, info = env.step(action)

        # the smaller theta and closer to center the better

        totalR += env_reward
        RL.store_memory(observation, action, env_reward, observation_)

        ep_r += env_reward
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
