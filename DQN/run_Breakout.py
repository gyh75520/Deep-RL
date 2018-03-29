import gym
from CNN_Brain_new import CNN_Brain as brain
from Agent import Agent
import numpy as np
from wrap_env import wrap_env
# import blosc
# import scipy.ndimage as ndimage

env = gym.make("{}NoFrameskip-v4".format('Breakout'))   # 定义使用 gym 库中的那一个环境
n_actions = env.action_space.n

print(env.action_space.n)  # 查看这个环境中可用的 action 有多少个
print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
# print(env.observation_space.high)   # 查看 observation 最高取值
# print(env.observation_space.low)    # 查看 observation 最低取值


Brain = brain(n_actions=env.action_space.n,
              observation_width=84,
              observation_height=84,
              observation_depth=4,
              filters_per_layer=np.array([32, 64, 64]),
              restore=False,
              output_graph=False,
              checkpoint_dir='Breakout_DQN_CNN_Net'
              )
RL = Agent(
    brain=Brain,
    n_actions=env.action_space.n,
    observation_space_shape=env.observation_space.shape,
    reward_decay=0.99,
    MAX_EPSILON=1,  # epsilon 的最大值
    MIN_EPSILON=0.1,  # epsilon 的最小值
    LAMBDA=0.000002,
    replace_target_iter=10000,
    memory_size=1000000,
    batch_size=32,
)


env = wrap_env(env)
total_steps = 0
learn_start_size = 50000

for i_episode in range(1, 10001):
    if i_episode % 1000 == 0:
        Brain.save()
    observation = env.reset()
    oneLife_totalR = 0
    done_times = 0
    LifeR = []
    while True:
        env.render()

        if len(RL.memory) > learn_start_size:
            action = RL.choose_action(observation)
        else:
            action = np.random.randint(0, n_actions)

        observation_, reward, done, info = env.step(action)
        clippedReward = min(1, max(-1, reward))
        # the smaller theta and closer to center the better

        oneLife_totalR += reward
        RL.store_memory(observation, action, clippedReward, observation_, done)

        if len(RL.memory) >= learn_start_size:
            if len(RL.memory) == learn_start_size:
                print('\n---------------------------------- Start Training ----------------------------------------')
            if total_steps % 4 == 0:
                RL.learn()
        if done:
            done_times += 1
            LifeR.append(oneLife_totalR)
            oneLife_totalR = 0
            if done_times == 5:
                print('episode: ', i_episode,
                      ' epsilon: ', round(RL.epsilon, 2),
                      'Each_Life_reward:', LifeR,
                      'total_reward:', sum(LifeR))
                break

        observation = observation_
        total_steps += 1


# Brain.save()
