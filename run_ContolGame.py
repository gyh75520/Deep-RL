import gym
import numpy as np
from utils import data_save4mlp
from config import pqExp_config as configs


def run_controlGame(episode, env, Agent, plt_q=False):
    # set_memory_with_random()
    for i_episode in range(episode):
        observation = env.reset()
        if plt_q:
            q_change = [observation]
            action_change = Agent.choose_action(observation)
        totalR = 0
        while True:
            # env.render()
            action = Agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            totalR += reward
            Agent.store_memory(observation, action, reward, observation_, done)
            # Agent.learn(costFlag=done) # costFlag = True 记录 loss
            Agent.learn()
            observation = observation_
            if done:
                print('episode: ', i_episode, ' epsilon: ', Agent.epsilon, 'total_reward:', totalR)
                if plt_q:
                    Agent.statistical_values(totalR, q_change, action_change)
                else:
                    Agent.statistical_values(totalR)
                break

    # Brain.save()  # 存储神经网络


def run_Pendulum(episode, env, Agent, plt_q=False):
    ACTION_SPACE = 11
    # set_memory_with_random()
    for i_episode in range(episode):
        observation = env.reset()
        if plt_q:
            q_change = [observation]
            action_change = Agent.choose_action(observation)
        totalR = 0
        while True:
            # env.render()
            action = Agent.choose_action(observation)
            f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)
            observation_, reward, done, info = env.step(np.array([f_action]))
            totalR += reward
            reward /= 10
            Agent.store_memory(observation, action, reward, observation_, done)
            # Agent.learn(costFlag=done) # costFlag = True 记录 loss
            Agent.learn()
            observation = observation_
            if done:
                print('episode: ', i_episode, ' epsilon: ', Agent.epsilon, 'total_reward:', totalR)
                if plt_q:
                    Agent.statistical_values(totalR, q_change, action_change)
                else:
                    Agent.statistical_values(totalR)
                break

    # Brain.save()  # 存储神经网络


def run_Game(model, env_name, episodes):
    if model == 'DQN':
        from model.mlpBrain import DQN_Brain as Brain
        from model.agent import DQN_Agent as Agent
    elif model == 'DDQN':
        from model.mlpBrain import DDQN_Brain as Brain
        from model.agent import DDQN_Agent as Agent
    elif model == 'PDQN':
        from model.mlpBrain import PDQN_Brain as Brain
        from model.agent import PDQN_Agent as Agent
    elif model == 'PDDQN':
        from model.mlpBrain import PDDQN_Brain as Brain
        from model.agent import PDDQN_Agent as Agent
    elif model == 'DQN_PER':
        from model.mlpBrain import DQN_PER_Brain as Brain
        from model.agent import DQN_PER_Agent as Agent
    elif model == 'DDQN_PER':
        from model.mlpBrain import DDQN_PER_Brain as Brain
        from model.agent import DDQN_PER_Agent as Agent
    elif model == 'DQN_InAday':
        from model.mlpBrain import DQN_InAday_Brain as Brain
        from model.agent import DQN_InAday_Agent as Agent
    elif model == 'DQN_PER_Ipm':
        from model.mlpBrain import DQN_PER_Ipm_Brain as Brain
        from model.agent import DQN_PER_Ipm_Agent as Agent
    elif model == 'DDQN_PER_Ipm':
        from model.mlpBrain import DDQN_PER_Ipm_Brain as Brain
        from model.agent import DDQN_PER_Ipm_Agent as Agent

    env = gym.make(env_name)   # 定义使用 gym 库中的那一个环境
    # env = env.unwrapped  # 注释掉的话 每局游戏 reward之和最高200

    n_actions = 11 if env_name == 'Pendulum-v0' else env.action_space.n

    print('\nThe config:\n', configs, '\n')
    neurons_per_layer = configs['Brain']['neurons_per_layer']
    learning_rate = configs['Brain']['learning_rate']
    output_graph = configs['Brain']['output_graph']
    restore = configs['Brain']['restore']

    reward_decay = configs['Agent']['reward_decay']
    replace_target_iter = configs['Agent']['replace_target_iter']
    memory_size = configs['Agent']['memory_size']
    batch_size = configs['Agent']['batch_size']
    MAX_EPSILON = configs['Agent']['MAX_EPSILON']
    MIN_EPSILON = configs['Agent']['MIN_EPSILON']
    LAMBDA = configs['Agent']['LAMBDA']

    # learning_rate 重要
    # restore 和 MAX_EPSILON 一起调整
    brain = Brain(
        n_actions=n_actions,
        n_features=env.observation_space.shape[0],
        neurons_per_layer=neurons_per_layer,
        learning_rate=learning_rate,
        output_graph=output_graph,
        restore=restore,
        checkpoint_dir=(env_name + '_' + model + '_MLP_Net')
    )
    agent = Agent(
        brain=brain,
        n_actions=n_actions,
        observation_space_shape=env.observation_space.shape,
        reward_decay=reward_decay,
        replace_target_iter=replace_target_iter,
        memory_size=memory_size,
        batch_size=batch_size,
        MAX_EPSILON=MAX_EPSILON,
        MIN_EPSILON=MIN_EPSILON,
        LAMBDA=LAMBDA,
    )

    if env_name == 'Pendulum-v0':
        run_Pendulum(episodes, env, agent, False)
    else:
        run_controlGame(episodes, env, agent, False)  # 4-th params = True 记录 q value
    data_save4mlp(model, env_name, brain, agent)
