import gym
from MLP import Neural_Networks as brain
from Agent import Agent
import numpy as np


env = gym.make('Acrobot-v1')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

Brain = brain(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    neurons_per_layer=np.array([8, 4]),
    learning_rate=0.01,
    output_graph=True)
RL = Agent(
    brain=Brain,
    n_actions=env.action_space.n,
    observation_space_shape=env.observation_space.shape,
    reward_decay=0.9,
    replace_target_iter=100,
    memory_size=2000,
)


total_steps = 0


for i_episode in range(20):

    observation = env.reset()
    ep_r = 0
    totalR = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        totalR += reward

        RL.store_memory(observation, action, reward, observation_, done)

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
