"""
This Memory code is modified version and the original code is from:
https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
"""
from .SumTree import SumTree
import numpy as np


class Memory(object):  # stored as ( s, a, r, s_ ,done) in SumTree

    def __init__(self, capacity):
        self.sumTree = SumTree(capacity)
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  # clipped abs error

    def getStoredSize(self):
        return self.sumTree.storedSize

    def store(self, transition):
        max_p = np.max(self.sumTree.tree[-self.sumTree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.sumTree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        tree_idx, batch_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,), dtype=object), np.empty((n,))
        pri_seg = self.sumTree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.sumTree.tree[self.sumTree.capacity - 1:self.sumTree.capacity + self.sumTree.storedSize - 1]) / \
            self.sumTree.total_p     # for later calculate ISweight
        # print('min_prob', min_prob, self.sumTree.tree[self.sumTree.capacity - 2:self.sumTree.capacity + self.sumTree.storedSize - 2])
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.sumTree.get_leaf(v)
            prob = p / self.sumTree.total_p
            ISWeights[i] = np.power(prob / min_prob, -self.beta)
            # print(idx, p, data)
            tree_idx[i], batch_memory[i] = idx, data
        return tree_idx, batch_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        # print('tree_idx', tree_idx, 'abs_errors', abs_errors, 'data_pointer', self.sumTree.data_pointer)
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.sumTree.update(ti, p)

    # add for own per improvement
    def batch_nearby_update(self, tree_idx):
        # print('ps', ps)
        ipmFactor = 1.2
        K = 2
        for k in range(1, K + 1):
            for ti in tree_idx + k:
                if (ti < 2 * self.sumTree.capacity - 1):
                    ps = self.sumTree.tree[ti]
                    ps_new = ps * ipmFactor
                    self.sumTree.update(ti, ps_new)

            for ti in tree_idx - k:
                if (ti > self.sumTree.capacity - 1):
                    ps = self.sumTree.tree[ti]
                    ps_new = ps * ipmFactor
                    self.sumTree.update(ti, ps_new)
