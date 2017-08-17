# -*- coding: UTF-8 -*-
from Brain import Brain as brain
from Agent import Agent
import numpy as np


Brain = brain(n_actions=3, n_features=4, learning_rate=0.00025, output_graph=True)
RL = Agent(
    brain=Brain,
    n_actions=3,
    observation_space_shape=(4,),
    reward_decay=0.9,
    replace_target_iter=100,
)

total_steps = 0


for i_episode in range(200):

    observation = np.array([1, 2, 3, 4])
    totalR = 0
    RL.set_policy_sigma()
    while True:
        # env.render()

        action = RL.choose_action(observation)
        print('choose_action', action)
        # observation_, reward, done, info = env.step(action)
        observation_, reward, done = np.array([2, 3, 5, 6]), 1, False
        # the smaller theta and closer to center the better
        if total_steps % 10 == 0 and total_steps != 0:
            done = True
            total_steps += 1
        totalR += reward
        RL.store_memory(observation, action, reward, observation_)
        # if total_steps > 1000:
        RL.learn()

        if done:
            print('episode: ', i_episode,
                  ' epsilon: ', round(RL.epsilon, 2),
                  'total_reward:', totalR)
            break

        observation = observation_
        total_steps += 1
RL.plot_cost()
