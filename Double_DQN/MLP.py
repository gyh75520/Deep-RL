# -*- coding: UTF-8 -*-
"""
by Howard
using:
- python: 3.6
- Tensorflow: 1.2.1
- gym: 0.9.2
"""

import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from Deep_Q_Learning.MLP import Neural_Networks as NN


class Neural_Networks(NN):
    def __init__(
        self,
        n_actions,  # 动作数，也就是输出层的神经元数
        n_features,  # 特征数，也就是输入的 vector 大小
        neurons_per_layer=np.array([10]),  # 隐藏层每层神经元数
        activation_function=tf.nn.relu,  # 激活函数
        Optimizer=tf.train.AdamOptimizer,  # 更新方法 tf.train.AdamOptimizer tf.train.RMSPropOptimizer GradientDescentOptimizer..
        learning_rate=0.01,  # 学习速率
        w_initializer=tf.random_normal_initializer(0., 0.3),
        b_initializer=tf.constant_initializer(0.1),
        output_graph=False,  # 使用 tensorboard
        restore=False,  # 是否使用存储的神经网络
        checkpoint_dir='Double_DQN_MLP_Net',  # 存储的dir name
    ):
        super(Neural_Networks, self).__init__(n_actions, n_features, neurons_per_layer,  activation_function, Optimizer,
                                              learning_rate,  w_initializer, b_initializer, output_graph, restore, checkpoint_dir)
