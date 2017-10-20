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


class CNN:
    def __init__(
        self,
        n_actions,  # 动作数，也就是输出层的神经元数
        observation_width,  # 图片的width
        observation_height,  # 图片的height
        observation_depth,  # 图片的depth
        filters_per_layer=np.array([32, 64]),  # Conv_and_Pool_Layer i 层中 卷积的filters
        activation_function=tf.nn.relu,  # 激活函数
        kernel_size=(5, 5),  # 卷积核的size
        conv_strides=(1, 1),  # 卷积层的strides
        padding='same',  # same or valid
        b_initializer=tf.zeros_initializer(),  # tf.constant_initializer(0.1)
        pooling_function=tf.layers.max_pooling2d,  # max_pooling2d or average_pooling2d
        pool_size=(2, 2),  # pooling的size
        pool_strides=(2, 2),  # pooling的strides
        Optimizer=tf.train.AdamOptimizer,  # 更新方法 tf.train.AdamOptimizer tf.train.RMSPropOptimizer..
        learning_rate=0.01,  # 学习速率
        output_graph=False,  # 使用 tensorboard
        restore=False,  # 是否使用存储的神经网络
        checkpoint_dir='My_CNN_Net',  # 存储的dir name
    ):
        self.n_actions = n_actions
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.observation_depth = observation_depth
        self.filters_per_layer = filters_per_layer
        self.activation_function = activation_function
        self.kernel_size = kernel_size
        self.conv_strides = conv_strides
        self.padding = padding
        self.b_initializer = b_initializer
        self.pooling_function = pooling_function
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.lr = learning_rate
        self.Optimizer = Optimizer
        self.b_initializer = b_initializer
        self.output_graph = output_graph
        self._build_net()
        self.checkpoint_dir = checkpoint_dir

        self.sess = tf.Session()

        if self.output_graph:
            if tf.gfile.Exists("graph/"):
                tf.gfile.DeleteRecursively("graph/")
            tf.gfile.MakeDirs("graph/")
            self.merged = tf.summary.merge_all()
            # terminal 输入 tensorboard --logdir graph
            self.writer = tf.summary.FileWriter("graph/", self.sess.graph)

        if restore:
            self.restore()
        else:
            self.sess.run(tf.global_variables_initializer())
        # self.cost_his = []

    def _build_net(self):
        def add_conv_and_pool_layer(
            inputs,
            n_layer,  # 当前层是第几层
            activation_function=tf.nn.relu,  # 激活函数
            filters=32,
            kernel_size=(5, 5),  # 卷积核的size
            conv_strides=(1, 1),  # 卷积层的strides
            padding='same',  # same or valid
            pooling_function=tf.layers.max_pooling2d,  # max_pooling2d or average_pooling2d
            pool_size=(2, 2),  # pooling的size
            pool_strides=(2, 2),  # pooling的strides
        ):
            layer_name = 'Conv_and_Pool_Layer%s' % n_layer
            with tf.variable_scope(layer_name):
                conv_name = 'Conv%s' % n_layer
                pool_name = 'Pool%s' % n_layer
                conv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, padding=padding, strides=conv_strides,
                                        activation=activation_function, bias_initializer=self.b_initializer, name=conv_name)
                pool = pooling_function(inputs=conv, pool_size=pool_size, strides=pool_strides, name=pool_name)
            return pool

        def build_layers(inputs, filters_per_layer):
            filters_per_layer = filters_per_layer.ravel()  # 平坦化数组
            layer_numbers = filters_per_layer.shape[0]
            l_range = range(0, layer_numbers)
            for l in l_range:  # 构造卷积和池化层
                filters = filters_per_layer[l]
                inputs = add_conv_and_pool_layer(inputs=inputs, n_layer=l + 1, activation_function=self.activation_function,
                                                 filters=filters, kernel_size=self.kernel_size, conv_strides=self.conv_strides,
                                                 padding=self.padding, pooling_function=self.pooling_function,
                                                 pool_size=self.pool_size, pool_strides=self.pool_strides)

            # 构造全连接层
            flat_size = inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
            inputs_flat = tf.reshape(inputs, [-1, int(flat_size)])
            dense = tf.layers.dense(inputs=inputs_flat, units=24, activation=tf.nn.relu)

            # 添加 dropout 处理过拟合
            dropout = tf.layers.dropout(inputs=dense, rate=0.4)
            out_size = self.n_actions
            # 输出层
            out = tf.layers.dense(inputs=dropout, units=out_size)
            return out

        # ------------------ 创建 eval 神经网络, 及时提升参数 ------------------
        self.s = tf.placeholder(tf.float32, [None, self.observation_width, self.observation_height, self.observation_depth], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            self.q_eval = build_layers(self.s, self.filters_per_layer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            if self.output_graph:
                tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('train'):
            self.train_op = self.Optimizer(self.lr).minimize(self.loss)

        # ------------------ 创建 target 神经网络, 提供 target Q ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.observation_width, self.observation_height, self.observation_depth], name='s_')
        with tf.variable_scope('target_net'):
            self.q_next = build_layers(self.s_, self.filters_per_layer)

    def train(self, input_s, q_target, learn_step_counter):
        # 训练 eval 神经网络
        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.s: input_s, self.q_target: q_target})
        return cost

    def output_tensorboard(self, input_s, q_target, input_s_, learn_step_counter):
        if self.output_graph:
            # 每隔100步记录一次
            if learn_step_counter % 100 == 0:
                rs = self.sess.run(self.merged, feed_dict={self.s: input_s, self.q_target: q_target, self.s_: input_s_})
                self.writer.add_summary(rs, learn_step_counter)

    def predict_eval_action(self, input_s):
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: input_s})
        return actions_value

    def predict_target_action(self, input_s):
        actions_value = self.sess.run(self.q_next, feed_dict={self.s_: input_s})
        return actions_value

    def replace_target_params(self):
        # 将 target_net 的参数 替换成 eval_net 的参数
        rr = re.compile('target_net')
        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, rr)
        rr = re.compile('eval_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, rr)
        for t, e in zip(t_params, e_params):
            self.sess.run(tf.assign(t, e))

    def save(self):
        # 存储神经网络
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.checkpoint_dir + "/save_net.ckpt")
        print("\nSave to path: ", save_path)

    def restore(self):
        # 使用存储的神经网络
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)  # ckpt.model_checkpoint_path表示模型存储的位置
            print('\nRestore Sucess')
        else:
            raise Exception("Check model_checkpoint_path Exist?")


if __name__ == '__main__':
    Brain = CNN(n_actions=2, observation_width=210, observation_height=160, observation_depth=3, output_graph=True)
