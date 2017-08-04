import gym
from MLP import Neural_Networks as brain
from Agent import Agent
import numpy as np


env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

Brain = brain(n_actions=env.action_space.n, n_features=env.observation_space.shape[0], neurons_per_layer=np.array([10]), output_graph=True)
RL = Agent(
    brain=Brain,
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    reward_decay=0.9,
    replace_target_iter=300,
    memory_size=3000,
    MIN_EPSILON=0.1,
)


total_steps = 0


for i_episode in range(10):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        position, velocity = observation_

        # the higher the better
        reward = abs(position - (-0.5))     # r in [0, 1]

        RL.store_memory(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward
        if done:
            get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', i_episode,
                  get,
                  '| Ep_r: ', round(ep_r, 4),
                  '| Epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
