# -*- coding: UTF-8 -*-
"""
by Howard
using:
- python: 3.6
"""


def cleanstring(string):
    return string.replace(".", "").replace("^", "").replace("-", "").replace(":", "").replace(" ", "_")


def mkdir(path):
    import os
    path = path.strip()  # 去除首位空格
    path = path.rstrip("\\")  # 去除尾部 \ 符号
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
    else:
        print(path + ' 目录已存在')


def data_save4mlp(algorithm_name, env_name, Brain, Agent):
    import datetime
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    filename = cleanstring(env_name) + "_" + algorithm_name + '_' + cleanstring(nowTime)
    SETUP = cleanstring(nowTime) + '|' + env_name + "|algorithm:" + algorithm_name + "|neurons_per_layer:" + str(Brain.neurons_per_layer) + \
        "|learning_rate:" + str(Brain.lr) + "|reward_decay:" + str(Agent.gamma) + \
        "|replace_target_iter:" + str(Agent.replace_target_iter) + "|memory_size:" + \
        str(Agent.memory_size) + "|batch_size:" + str(Agent.batch_size) + "|MAX_EPSILON:" + \
        str(Agent.MAX_EPSILON) + "|MIN_EPSILON:" + str(Agent.MIN_EPSILON) + "|LAMBDA:" + str(Agent.LAMBDA)

    outStr = "# Experiment_SETUP:\"" + SETUP + "\""
    # outStr += "\nimport pylab\n"
    # rewards
    outStr += "\nRewards = " + str(Agent.rewards) + "\n"
    # cost_his
    outStr += "\ncost_his = " + str(Agent.cost_his) + "\n"
    # q value
    if hasattr(Agent, 'q_change_list'):
        outStr += "\nq_change_list = " + str(Agent.q_change_list) + "\n"

    path = cleanstring(env_name) + algorithm_name + "_data"
    mkdir(path)
    f = open(path + "/" + filename + ".py", "w")
    f.write(outStr)
    f.close()

    print("\nsave to ", path, "successful!")


def data_save4cnn(algorithm_name, env_name, Brain, Agent):
    import datetime
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    filename = cleanstring(nowTime) + '_' + env_name + "_" + algorithm_name
    SETUP = cleanstring(nowTime) + '|' + env_name + "|algorithm:" + algorithm_name + \
        "|filters_per_layer:" + str(Brain.filters_per_layer) + "|kernel_size_per_layer:" + \
        str(Brain.kernel_size_per_layer) + "|conv_strides_per_layer:" + str(Brain.conv_strides_per_layer) +\
        "|learning_rate:" + str(Brain.lr) + "|reward_decay:" + str(Agent.gamma) + \
        "|replace_target_iter:" + str(Agent.replace_target_iter) + "|memory_size:" + \
        str(Agent.memory_size) + "|batch_size:" + str(Agent.batch_size) + "|MAX_EPSILON:" + \
        str(Agent.MAX_EPSILON) + "|MIN_EPSILON:" + str(Agent.MIN_EPSILON) + "|LAMBDA:" + str(Agent.LAMBDA)

    outStr = "Experiment_SETUP:\"" + SETUP + "\""
    outStr += "\nimport pylab\n"
    # rewards
    outStr += "\nRewards = " + str(Agent.rewards) + "\n"
    # cost_his
    outStr += "\ncost_his = " + str(Agent.cost_his) + "\n"
    # q value
    if hasattr(Agent, 'q_change_list'):
        outStr += "\nq_change_list = " + str(Agent.q_change_list) + "\n"

    path = env_name + algorithm_name + "_data"
    mkdir(path)
    f = open(path + "/" + filename + ".py", "w")
    f.write(outStr)
    f.close()

    print("\nsave to ", path, "successful!")


def plot(title, datas, data_labels, ylabel):
    import matplotlib.pyplot as plt
    for data, data_label in zip(datas, data_labels):
        plt.plot(range(len(data)), data, label=data_label)

    plt.legend(loc='best')
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()


def get_files(dir):
    import os
    for root, dirs, files in os.walk(dir):
        return files


def get_rewards(dir):
    import numpy as np
    files = get_files(dir)
    Rewards_list = []

    Rewards_list = []
    for file in files:
        file_name = file.replace(".py", '')
        str = "from %s.%s import Rewards" % (dir, file_name)
        str += '\nRewards_list.append(Rewards)'
        # print(str)
        exec(str)

    Rewards_total = np.zeros(len(Rewards_list[0]))
    for Rewards in Rewards_list:
        Rewards = np.array(Rewards)
        Rewards_total += Rewards
    Rewards_mean = Rewards_total / len(Rewards_list)
    return Rewards_mean


if __name__ == '__main__':

    datas = [Rewards_1, Rewards_2]
    data_labels = ['ddqnPer', 'ddqn']
    ylabel = 'reward'
    plot("title", datas, data_labels, ylabel)
