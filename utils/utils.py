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


def writeData2File(algorithm_name, env_name, Brain, Agent, DataStorage):
    import datetime
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    filename = cleanstring(env_name) + "_" + algorithm_name + '_' + cleanstring(nowTime)
    SETUP = cleanstring(nowTime) + '|' + env_name + "|algorithm:" + algorithm_name + "\n"

    brainDict = Brain.__dict__
    agentDict = Agent.__dict__
    dataStorageDict = DataStorage.__dict__

    SETUP += "'''Brain\n"
    for k, v in brainDict.items():
        SETUP += str(k) + ":" + str(v) + "\n"
    SETUP += "'''\n"

    SETUP += "'''Agent\n"
    for k, v in agentDict.items():
        if k not in ['memory']:
            SETUP += str(k) + ":" + str(v) + "\n"
    SETUP += "'''\n"

    # SETUP += "# Brain:" + str(Brain.__dict__) + "\n"
    # SETUP += "# DataStorage:" + str(DataStorage.__dict__) + "\n"
    outStr = "# Experiment_SETUP:\"" + SETUP + "\n"
    # outStr += "\nimport pylab\n"
    # rewards
    for k, v in dataStorageDict.items():
        outStr += "\n" + str(k) + " = " + str(v) + "\n"
    # outStr += "\nRewards = " + str(DataStorage.rewards) + "\n"

    path = cleanstring(env_name) + algorithm_name + "_data"
    mkdir(path)
    f = open(path + "/" + filename + ".py", "w")
    f.write(outStr)
    f.close()

    print("\nsave to ", path, "successful!")


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
    filename = cleanstring(env_name) + "_" + algorithm_name + '_' + cleanstring(nowTime)
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

    path = cleanstring(env_name) + algorithm_name + "_data"
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


def get_Q(dir):
    import numpy as np
    files = get_files(dir)
    Q_list = []

    Q_list = []
    for file in files:
        file_name = file.replace(".py", '')
        # str = "from %s.%s import q_change_list" % (dir, file_name)
        # str += '\nQ_list.append(q_change_list)'
        str = "from %s.%s import Q_value" % (dir, file_name)
        str += '\nQ_list.append(Q_value)'
        # print(str)
        exec(str)

    Q_total = np.zeros(len(Q_list[0]))
    for Q in Q_list:
        Q = np.array(Q)
        Q_total += Q
    Q_mean = Q_total / len(Q_list)
    return Q_mean


def linearSmooth3(inputs):
    import numpy as np
    print(inputs[0: 10])
    N = inputs.shape[0]
    out = np.zeros(N)
    out[0] = (5.0 * inputs[0] + 2.0 * inputs[1] - inputs[2]) / 6.0
    i = 1
    while i < (N - 1):
        out[i] = (inputs[i - 1] + inputs[i] + inputs[i + 1]) / 3.0
        i += 1

    out[N - 1] = (5.0 * inputs[N - 1] + 2.0 * inputs[N - 2] - inputs[N - 3]) / 6.0

    # print(out[0: 10])
    return out


if __name__ == '__main__':
    import sys
    sys.path.append("..")
    Rewards = get_rewards('Acrobotv1DQN_PER_Ipm_data')
    datas = [Rewards]
    data_labels = ['dqn']
    ylabel = 'reward'
    plot("title", datas, data_labels, ylabel)
