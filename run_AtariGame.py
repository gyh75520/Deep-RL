import gym
from utils import writeData2File, DataStorage, wrap_env
from config import cnnExp_config as configs
import numpy as np


def run_AtariGame(episode, model, env, env_name, lifes, Agent, Brain, DataStorage, replay_start_size, update_frequency, is_storeQ=False):
    total_steps = 0
    learn_start_size = replay_start_size
    for i_episode in range(1, episode):
        if i_episode % 5000 == 0:
            Brain.save()
            writeData2File(model, env_name, Brain, Agent, DataStorage)
        observation = env.reset()
        if is_storeQ:
            InitState = [observation]
            FirstAction = Agent.choose_action(observation)
        oneLife_totalR = 0
        done_times = 0
        LifeR = []
        while True:
            # env.render()

            if Agent.getStoredSize() > learn_start_size:
                action = Agent.choose_action(observation)
            else:
                action = np.random.randint(0, env.action_space.n)

            observation_, reward, done, info = env.step(action)
            clippedReward = min(1, max(-1, reward))
            # the smaller theta and closer to center the better

            oneLife_totalR += reward
            Agent.store_memory(observation, action, clippedReward, observation_, done)

            if Agent.getStoredSize() >= learn_start_size:
                if Agent.getStoredSize() == learn_start_size:
                    print('\n---------------------------------- Start Training ----------------------------------------')
                if total_steps % update_frequency == 0:
                    Agent.learn()
            if done:
                done_times += 1
                LifeR.append(oneLife_totalR)
                oneLife_totalR = 0
                if done_times == lifes:
                    if is_storeQ:
                        Q_value = Agent.brain.predict_eval_action(InitState)
                        DataStorage.store_Q(Q_value[0][FirstAction])
                    DataStorage.store_reward(sum(LifeR))
                    print('episode: ', i_episode, ' epsilon: ', round(Agent.epsilon, 2), 'Each_Life_reward:', LifeR, 'total_reward:', sum(LifeR))
                    break

            observation = observation_
            total_steps += 1


def run_Game(model, env_name, lifes, episodes):
    if model == 'DQN':
        from model.cnnBrain import DQN_Brain as Brain
        from model.agent import DQN_Agent as Agent
    elif model == 'DDQN':
        from model.cnnBrain import DDQN_Brain as Brain
        from model.agent import DDQN_Agent as Agent
    elif model == 'PDQN':
        from model.cnnBrain import PDQN_Brain as Brain
        from model.agent import PDQN_Agent as Agent
    elif model == 'PDDQN':
        from model.cnnBrain import PDDQN_Brain as Brain
        from model.agent import PDDQN_Agent as Agent
    elif model == 'DQN_PER':
        from model.cnnBrain import DQN_PER_Brain as Brain
        from model.agent import DQN_PER_Agent as Agent
    elif model == 'DDQN_PER':
        from model.cnnBrain import DDQN_PER_Brain as Brain
        from model.agent import DDQN_PER_Agent as Agent

    # lifes = 5
    # env_name = 'Breakout'
    env = gym.make("{}NoFrameskip-v4".format(env_name))   # 定义使用 gym 库中的那一个环境

    print('\nThe config:\n', configs, '\n')
    filters_per_layer = configs['Brain']['filters_per_layer']
    kernel_size_per_layer = configs['Brain']['kernel_size_per_layer']
    conv_strides_per_layer = configs['Brain']['conv_strides_per_layer']
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

    replay_start_size = configs['ExperienceReplay']['replay_start_size']
    update_frequency = configs['ExperienceReplay']['update_frequency']

    brain = Brain(n_actions=env.action_space.n,
                  observation_width=84,
                  observation_height=84,
                  observation_depth=4,
                  learning_rate=learning_rate,
                  filters_per_layer=filters_per_layer,
                  kernel_size_per_layer=kernel_size_per_layer,
                  conv_strides_per_layer=conv_strides_per_layer,
                  restore=restore,
                  output_graph=output_graph,
                  checkpoint_dir=(env_name + '_' + model + '_CNN_Net')
                  )
    agent = Agent(
        brain=brain,
        n_actions=env.action_space.n,
        observation_space_shape=env.observation_space.shape,
        reward_decay=reward_decay,
        MAX_EPSILON=MAX_EPSILON,  # epsilon 的最大值
        MIN_EPSILON=MIN_EPSILON,  # epsilon 的最小值
        LAMBDA=LAMBDA,
        replace_target_iter=replace_target_iter,
        memory_size=memory_size,
        batch_size=batch_size,
    )
    dataStorage = DataStorage()

    env = wrap_env(env)
    run_AtariGame(episodes, model, env, env_name, lifes, agent, brain, dataStorage, replay_start_size, update_frequency, False)  # last params = True 记录 q value
    # writeData2File(model, env_name, brain, agent, dataStorage)
