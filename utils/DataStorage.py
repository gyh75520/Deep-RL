# -*- coding: UTF-8 -*-
"""
by Howard
using:
- python: 3.6

"""


class DataStorage:

    def store_reward(self, reward):
        if not hasattr(self, 'Rewards'):
            self.Rewards = []
        self.Rewards.append(reward)

    def store_Q(self, Q):
        if not hasattr(self, 'Q_value'):
            self.Q_value = []
        self.Q_value.append(Q)


if __name__ == '__main__':
    data = DataStorage()
    data.store_reward(2)
