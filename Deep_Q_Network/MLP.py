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


class Neural_Networks:
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
        checkpoint_dir='My_MLP_Net',  # 存储的dir name
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.neurons_per_layer = neurons_per_layer
        self.activation_function = activation_function
        self.lr = learning_rate
        self.Optimizer = Optimizer
        self.w_initializer = w_initializer
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
        def add_layer(
            inputs,
            in_size,  # 传入数，即上一层神经元数
            out_size,  # 当前层的神经元数
            n_layer,  # 当前层是第几层
            c_names,
            activation_function=None,  # 激活函数
        ):
            layer_name = 'Layer%s' % n_layer
            with tf.variable_scope(layer_name):
                Weights = tf.get_variable('Weights', [in_size, out_size], initializer=self.w_initializer, collections=c_names)
                biases = tf.get_variable('biases', [1, out_size], initializer=self.b_initializer, collections=c_names)
                Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

                if activation_function is None:
                    outputs = Wx_plus_b
                else:
                    outputs = activation_function(Wx_plus_b, )

                if self.output_graph:
                    tf.summary.histogram(layer_name + '/weights', Weights)
                    tf.summary.histogram(layer_name + '/biases', biases)
                    tf.summary.histogram(layer_name + '/outputs', outputs)
            return outputs

        def build_layers(inputs, neurons_per_layer, c_names):
            neurons_per_layer = neurons_per_layer.ravel()  # 平坦化数组
            layer_numbers = neurons_per_layer.shape[0]  # 隐藏层层数
            neurons_range = range(0, layer_numbers)
            in_size = self.n_features
            for n_neurons in neurons_range:  # 构造隐藏层
                out_size = neurons_per_layer[n_neurons]
                inputs = add_layer(inputs, in_size, out_size, n_neurons + 1, c_names, self.activation_function)
                in_size = out_size

            out_size = self.n_actions
            out = add_layer(inputs, in_size, out_size, layer_numbers + 1, c_names, None)  # 构造输出层
            return out

        # ------------------ 创建 eval 神经网络, 及时提升参数 ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_eval = build_layers(self.s, self.neurons_per_layer, c_names)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            if self.output_graph:
                tf.summary.scalar('loss', self.loss)

        with tf.variable_scope('train'):
            self.train_op = self.Optimizer(self.lr).minimize(self.loss)

        # ------------------ 创建 target 神经网络, 提供 target Q ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, self.neurons_per_layer, c_names)

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
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
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
    Brain = Neural_Networks(n_actions=1, n_features=1, output_graph=True)
