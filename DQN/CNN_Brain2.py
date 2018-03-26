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
import re
from base_Brain import Brain


class CNN_Brain(Brain):
    def __init__(
        self,
        n_actions,  # 动作数，也就是输出层的神经元数
        observation_width,  # 图片的width
        observation_height,  # 图片的height
        observation_depth,  # 图片的depth
        filters_per_layer=np.array([32, 64, 64]),  # Conv_and_Pool_Layer i 层中 卷积的filters
        activation_function=tf.nn.relu,  # 激活函数
        kernel_size_per_layer=[(8, 8), (4, 4), (3, 3)],  # 卷积核的size
        conv_strides_per_layer=[(4, 4), (2, 2), (1, 1)],  # 卷积层的strides
        padding='valid',  # same or valid
        b_initializer=tf.constant_initializer(0.01),  # tf.constant_initializer(0.1)
        pooling_function=tf.layers.max_pooling2d,  # max_pooling2d or average_pooling2d
        pool_size=(2, 2),  # pooling的size
        pool_strides=(2, 2),  # pooling的strides
        Optimizer=tf.train.RMSPropOptimizer,  # 更新方法 tf.train.AdamOptimizer tf.train.RMSPropOptimizer..
        learning_rate=0.00025,  # 学习速率
        output_graph=False,  # 使用 tensorboard
        restore=False,  # 是否使用存储的神经网络
        checkpoint_dir='DQN_CNN_Net',  # 存储的dir name
    ):
        # self.n_actions = n_actions
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.observation_depth = observation_depth
        self.filters_per_layer = filters_per_layer
        # self.activation_function = activation_function
        self.kernel_size_per_layer = kernel_size_per_layer
        self.conv_strides_per_layer = conv_strides_per_layer
        self.padding = padding
        self.b_initializer = b_initializer
        self.pooling_function = pooling_function
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.normalizeWeights = True
        # self.lr = learning_rate
        # self.Optimizer = Optimizer
        # self.output_graph = output_graph
        # self._build_net()
        # self.checkpoint_dir = checkpoint_dir
        # self.sess = tf.Session()
        super(CNN_Brain, self).__init__(n_actions, activation_function, Optimizer, learning_rate,  output_graph, restore, checkpoint_dir)

    def makeLayerVariables(self, shape, trainable, name_suffix):
        import math
        if self.normalizeWeights:
            # This is my best guess at what DeepMind does via torch's Linear.lua and SpatialConvolution.lua (see reset methods).
            # np.prod(shape[0:-1]) is attempting to get the total inputs to each node
            stdv = 1.0 / math.sqrt(np.prod(shape[0:-1]))
            weights = tf.Variable(tf.random_uniform(shape, minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + name_suffix)
            biases = tf.Variable(tf.random_uniform([shape[-1]], minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + name_suffix)
        else:
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), trainable=trainable, name='W_' + name_suffix)
            biases = tf.Variable(tf.fill([shape[-1]], 0.1), trainable=trainable, name='W_' + name_suffix)
        return weights, biases

    def _build_net(self):
        def build_layers(inputs, name, trainable=True,):
            # Second layer convolves 32 8x8 filters with stride 4 with relu
            with tf.variable_scope("cnn1_" + name):
                W_conv1, b_conv1 = self.makeLayerVariables([8, 8, 4, 32], trainable, "conv1")

                h_conv1 = tf.nn.relu(tf.nn.conv2d(inputs, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
                print(h_conv1)

            # Third layer convolves 64 4x4 filters with stride 2 with relu
            with tf.variable_scope("cnn2_" + name):
                W_conv2, b_conv2 = self.makeLayerVariables([4, 4, 32, 64], trainable, "conv2")

                h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
                print(h_conv2)

            # Fourth layer convolves 64 3x3 filters with stride 1 with relu
            with tf.variable_scope("cnn3_" + name):
                W_conv3, b_conv3 = self.makeLayerVariables([3, 3, 64, 64], trainable, "conv3")

                h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3, name="h_conv3")
                print(h_conv3)

            h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 64], name="h_conv3_flat")
            print(h_conv3_flat)

            # Fifth layer is fully connected with 512 relu units
            with tf.variable_scope("fc1_" + name):
                W_fc1, b_fc1 = self.makeLayerVariables([7 * 7 * 64, 512], trainable, "fc1")

                h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")
                print(h_fc1)

            # Sixth (Output) layer is fully connected linear layer
            with tf.variable_scope("fc2_" + name):
                W_fc2, b_fc2 = self.makeLayerVariables([512, self.n_actions], trainable, "fc2")

                y = tf.matmul(h_fc1, W_fc2) + b_fc2
                print(y)

            return y

        # ------------------ 创建 eval 神经网络, 及时提升参数 ------------------
        self.s = tf.placeholder(tf.uint8, [None, self.observation_width, self.observation_height, self.observation_depth], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')  # for calculating loss

        self.normalized_s = tf.to_float(self.s) / 255.0
        with tf.variable_scope('eval_net'):
            self.q_eval = build_layers(self.normalized_s, 'eval')

        with tf.variable_scope('loss'):
            self.action = tf.placeholder(tf.float32, [None, self.n_actions], name='action')  # one hot presentatio
            Q_action = tf.reduce_sum(tf.multiply(self.q_eval, self.action), reduction_indices=1)

            # --------new---------
            difference = tf.abs(Q_action - self.q_target)
            quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
            linear_part = difference - quadratic_part
            errors = (0.5 * tf.square(quadratic_part)) + linear_part
            self.loss = tf.reduce_sum(errors)
            # ----------------------
            # self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, Q_action))
            # self.loss = tf.reduce_mean(tf.square(self.q_target - Q_action))
            if self.output_graph:
                tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('train'):
            self.train_op = self.Optimizer(learning_rate=self.lr, decay=.95, epsilon=.01).minimize(self.loss)

        # ------------------ 创建 target 神经网络, 提供 target Q ------------------
        self.s_ = tf.placeholder(tf.uint8, [None, self.observation_width, self.observation_height, self.observation_depth], name='s_')
        self.normalized_s_ = tf.to_float(self.s_) / 255.0
        with tf.variable_scope('target_net'):
            self.q_next = build_layers(self.normalized_s_, 'target')

        # ------------------ replace_target_params op ------------------
        self.update_target = []
        rr = re.compile('target_net')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, rr)
        rr = re.compile('eval_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, rr)
        for t, e in zip(t_params, e_params):
            self.update_target.append(tf.assign(t, e))

    def replace_target_params(self):
        # 将 target_net 的参数 替换成 eval_net 的参数
        '''
        rr = re.compile('target_net')
        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, rr)
        rr = re.compile('eval_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, rr)
        for t, e in zip(t_params, e_params):
            self.sess.run(tf.assign(t, e))
        '''
        # 改成 op 速度更快
        self.sess.run(self.update_target)


if __name__ == '__main__':
    Brain = CNN(n_actions=2, observation_width=210, observation_height=160, observation_depth=3, output_graph=True)
