import gym
from CNN_Brain2 import CNN_Brain as brain
from Agent import Agent
import numpy as np
from video_env import video_env
# import blosc
import scipy.ndimage as ndimage
import scipy

env = gym.make('Breakout-v0')   # 定义使用 gym 库中的那一个环境
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
              checkpoint_dir='DQN_CNN_3_19_20_Net'
              )
RL = Agent(
    brain=Brain,
    n_actions=env.action_space.n,
    observation_space_shape=env.observation_space.shape,
    reward_decay=0.99,
    MAX_EPSILON=1,  # epsilon 的最大值
    MIN_EPSILON=0.1,  # epsilon 的最小值
    LAMBDA=0.00001,
    replace_target_iter=10000,
    memory_size=1000000,
    batch_size=32,
)


def concatenate(screens):
    return np.concatenate(screens, axis=2)


def preprocess(screen):
    screen = np.dot(screen, np.array([.299, .587, .114])).astype(np.uint8)
    screen = ndimage.zoom(screen, (0.4, 0.525))
    # print('screen', screen.shape)
    screen.resize((84, 84, 1))
    return screen


def processImage(img):
    rgb = scipy.misc.imresize(img, (84, 84), interp='bilinear')
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b     # extract luminance

    o = gray.astype('float32') / 128 - 1    # normalize
    o.resize((84, 84, 1))
    return o


env = video_env(env)
total_steps = 0
learn_start_size = 50000

for i_episode in range(100000):

    observation = env.reset()
    observation = processImage(observation)

    ep_r = 0
    totalR = 0
    observation_input = [observation, observation, observation, observation]
    state = concatenate(observation_input)
    while True:
        # env.render()

        if len(RL.memory) > learn_start_size:
            action = RL.choose_action(state)
        else:
            action = np.random.randint(0, n_actions)
        observation_, reward, done, info = env.step(action)

        totalR += reward

        if (len(observation_input) == 4):

            observation_ = processImage(observation_)
            observation_input.insert(0, observation_)
            observation_input.pop(4)
            state_ = concatenate(observation_input)
            # print(state.shape)
            # clipped all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged.
            clippedReward = min(1, max(-1, reward))
            RL.store_memory(state, action, clippedReward, state_, done)

        if len(RL.memory) > learn_start_size:
            if len(RL.memory) == learn_start_size + 1:
                print('\n---------------------------------- Start Training ----------------------------------------')
            RL.learn()

        state = state_
        total_steps += 1

        if done:
            print('episode: ', i_episode,
                  ' epsilon: ', round(RL.epsilon, 2),
                  'total_reward:', totalR)
            break


Brain.save()
