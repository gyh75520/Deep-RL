# DeepRL
Modularized implementation of popular deep RL algorithms by Tensorflow. My principal here is to reuse as much components as possible through different algorithms and switch easily between classical control tasks like CartPole and Atari games with raw pixel inputs.

Implemented algorithms:
* Deep Q-Learning (DQN)
* Double Deep Q-Learning (DDQN)
* Deep Q-Learning + PrioritizedExperienceReplay(DQN_PER)
* Double Deep Q-Learning + PrioritizedExperienceReplay(DDQN_PER)
* Deep Q-Learning + In A Day(DQN_InAday)


# Dependency
> Tested in macOS 10.13
* OpenAI gym
* Tensorflow v1.2.1
* Python  3.6



# Usage

```main.py``` contains examples for all the implemented algorithms

# References
* [Human Level Control through Deep Reinforcement Learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
* [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
* [Learning to Play in a Day: Faster Deep Reinforcement Learning by Optimality Tightening](https://arxiv.org/abs/1611.016062)
