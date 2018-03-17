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


class Brain:
    def __init__(
        self,
        n_actions,  # 动作数，也就是输出层的神经元数
        activation_function=tf.nn.relu,  # 激活函数
        Optimizer=tf.train.AdamOptimizer,  # 更新方法 tf.train.AdamOptimizer tf.train.RMSPropOptimizer GradientDescentOptimizer..
        learning_rate=0.01,  # 学习速率
        output_graph=False,  # 使用 tensorboard
        restore=False,  # 是否使用存储的神经网络
        checkpoint_dir='NN_MLP_Net',  # 存储的dir name
    ):
        self.n_actions = n_actions
        self.activation_function = activation_function
        self.lr = learning_rate
        self.Optimizer = Optimizer
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
            self.sess.run(self.update_target)  # is this necessary?
        # self.cost_his = []

    def _build_net(self):
        print('add your Neural_Networks')

    # 训练 eval 神经网络
    def train(self, input_s, q_target, action, learn_step_counter):
        self.sess.run(self.train_op, feed_dict={self.s: input_s, self.action: action, self.q_target: q_target})

    def output_tensorboard(self, input_s, q_target, input_s_, action, learn_step_counter):
        if self.output_graph:
            # 每隔100步记录一次
            if learn_step_counter % 100 == 0:
                rs = self.sess.run(self.merged, feed_dict={self.s: input_s, self.q_target: q_target, self.s_: input_s_, self.action: action})
                self.writer.add_summary(rs, learn_step_counter)

    def predict_eval_action(self, input_s):
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: input_s})
        return actions_value

    def predict_target_action(self, input_s):
        actions_value = self.sess.run(self.q_next, feed_dict={self.s_: input_s})
        return actions_value

    # 将 target_net 的参数 替换成 eval_net 的参数
    def replace_target_params(self):
        print('add your replace_target_params')

    # 存储神经网络
    def save(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.checkpoint_dir + "/save_net.ckpt")
        print("\nSave to path: ", save_path)

    # 使用存储的神经网络
    def restore(self):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(self.sess, ckpt.model_checkpoint_path)  # ckpt.model_checkpoint_path表示模型存储的位置
            print('\nRestore Sucess')
        else:
            raise Exception("Check model_checkpoint_path Exist?")
