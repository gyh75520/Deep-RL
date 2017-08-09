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
import matplotlib.pyplot as plt


class CNN:
    def __init__(
        self,
        n_actions,  # 动作数，也就是输出层的神经元数
        n_features_width,  # 特征数，也就是输入的矩阵的列数
        n_features_height,
        n_features_depth,
        neurons_per_layer=np.array([32, 64]),  # 隐藏层每层神经元数
        activation_function=tf.nn.relu,  # 激活函数
        Optimizer=tf.train.AdamOptimizer,  # 更新方法 tf.train.AdamOptimizer tf.train.RMSPropOptimizer..
        learning_rate=0.01,  # 学习速率
        w_initializer=tf.random_normal_initializer(0., 0.3),
        b_initializer=tf.constant_initializer(0.1),
        output_graph=False,  # 使用tensorboard
    ):
        self.n_actions = n_actions
        self.n_features_width = n_features_width
        self.n_features_height = n_features_height
        self.n_features_depth = n_features_depth
        self.neurons_per_layer = neurons_per_layer
        self.activation_function = activation_function
        self.lr = learning_rate
        self.Optimizer = Optimizer
        self.w_initializer = w_initializer
        self.b_initializer = b_initializer
        self.output_graph = output_graph
        self._build_net()

        self.sess = tf.Session()

        if self.output_graph:
            if tf.gfile.Exists("graph/"):
                tf.gfile.DeleteRecursively("graph/")
            tf.gfile.MakeDirs("graph/")
            self.merged = tf.summary.merge_all()
            # terminal 输入 tensorboard --logdir graph
            self.writer = tf.summary.FileWriter("graph/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        # self.cost_his = []

    def _build_net(self):
        def add_conv_and_pool_layer(
            inputs,
            n_layer,  # 当前层是第几层
            c_names,
            activation_function=tf.nn.relu,  # 激活函数
            filters=32,
            kernel_size=(5, 5),
            conv_strides=(1, 1),
            padding='same',
            bias_initializer=tf.zeros_initializer(),
            pooling_function=tf.layers.max_pooling2d,
            pool_size=(2, 2),
            pool_strides=(2, 2),
        ):
            layer_name = 'Conv_and_Pool_Layer%s' % n_layer
            with tf.variable_scope(layer_name):
                conv_name = 'Conv%s' % n_layer
                pool_name = 'Pool%s' % n_layer
                conv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding=padding, strides=conv_strides,
                                        activation=activation_function, bias_initializer=bias_initializer, name=conv_name)
                pool = pooling_function(inputs=conv, pool_size=pool_size, strides=pool_strides, name=pool_name)
            return pool

        def build_layers(inputs, neurons_per_layer, c_names):
            neurons_per_layer = neurons_per_layer.ravel()  # 平坦化数组
            layer_numbers = neurons_per_layer.shape[0]  # 隐藏层层数
            neurons_range = range(0, layer_numbers)
            for n_neurons in neurons_range:  # 构造隐藏层
                filters = neurons_per_layer[n_neurons]
                # inputs = add_layer(inputs, in_size, out_size, n_neurons + 1, c_names, self.activation_function)
                inputs = add_conv_and_pool_layer(inputs=inputs, n_layer=n_neurons + 1, c_names=c_names, filters=filters)
            print('\ninputs.shape', inputs.shape[1])
            flat_size = inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
            inputs_flat = tf.reshape(inputs, [-1, int(flat_size)])
            dense = tf.layers.dense(inputs=inputs_flat, units=24, activation=tf.nn.relu)

            # Add dropout operation; 0.6 probability that element will be kept
            dropout = tf.layers.dropout(inputs=dense, rate=0.4)
            out_size = self.n_actions
            out = tf.layers.dense(inputs=dropout, units=out_size)
            return out

        # ------------------ 创建 eval 神经网络, 及时提升参数 ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features_width, self.n_features_height, self.n_features_depth], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names = 'eval_net_params'
            self.q_eval = build_layers(self.s, self.neurons_per_layer, c_names)
            print(tf.get_collection('eval_net_params'))

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            if self.output_graph:
                tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('train'):
            self.train_op = self.Optimizer(self.lr).minimize(self.loss)

        # ------------------ 创建 target 神经网络, 提供 target Q ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features_width, self.n_features_height, self.n_features_depth], name='s_')
        with tf.variable_scope('target_net'):
            c_names = 'target_net_params'
            self.q_next = build_layers(self.s_, self.neurons_per_layer, c_names)

    def train(self, input_s, q_target, learn_step_counter):

        # 训练 eval 神经网络
        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.s: input_s, self.q_target: q_target})

        if self.output_graph:
            # 每隔100步记录一次
            if learn_step_counter % 100 == 0:
                rs = self.sess.run(self.merged, feed_dict={self.s: input_s, self.q_target: q_target, self.s_: input_s})
                self.writer.add_summary(rs, learn_step_counter)
        return cost

    def predict_eval_action(self, input_s):
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: input_s})
        return actions_value

    def predict_target_action(self, input_s):
        actions_value = self.sess.run(self.q_next, feed_dict={self.s_: input_s})
        return actions_value

    def replace_target_params(self):
        # 将 target_net 的参数 替换成 eval_net 的参数
        rr = re.compile('target_net')
        print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, rr)
        rr = re.compile('eval_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, rr)
        for t, e in zip(t_params, e_params):
            self.sess.run(tf.assign(t, e))


if __name__ == '__main__':
    Brain = CNN(n_actions=2, n_features_width=210, n_features_height=160, n_features_depth=3, output_graph=True)
