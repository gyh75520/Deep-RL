import gym
from CNN_Brain_new import CNN_Brain as Brain
from Agent import Agent
import numpy as np
from wrap_env import wrap_env
import sys
sys.path.append("..")
from utils import data_save4cnn
from cnnExp_config import configs

lifes = 5
env_name = 'Breakout'
env = gym.make("{}NoFrameskip-v4".format(env_name))   # 定义使用 gym 库中的那一个环境
env.seed(1)
print(env.action_space.sample())  # 查看这个环境中可用的 action 有多少个
print(env.observation_space.shape)    # 查看这个环境中可用的 state 的 observation 有多少个
print(env.observation_space.high)   # 查看 observation 最高取值
print(env.observation_space.low)    # 查看 observation 最低取值

n_actions = env.action_space.n
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
              checkpoint_dir=(env_name + '_DQN_CNN_Net')
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


env = wrap_env(env)


def run_game(episode, env, Agent, plt_q=False):
    total_steps = 0
    learn_start_size = replay_start_size
    for i_episode in range(1, episode):
        if i_episode % 1000 == 0:
            Brain.save()
        observation = env.reset()
        if plt_q:
            q_change = [observation]
            action_change = Agent.choose_action(observation)
        oneLife_totalR = 0
        done_times = 0
        LifeR = []
        while True:
            # env.render()

            if len(Agent.memory) > learn_start_size:
                action = Agent.choose_action(observation)
            else:
                action = np.random.randint(0, n_actions)

            observation_, reward, done, info = env.step(action)
            clippedReward = min(1, max(-1, reward))
            # the smaller theta and closer to center the better

            oneLife_totalR += reward
            Agent.store_memory(observation, action, clippedReward, observation_, done)

            if len(Agent.memory) >= learn_start_size:
                if len(Agent.memory) == learn_start_size:
                    print('\n---------------------------------- Start Training ----------------------------------------')
                if total_steps % update_frequency == 0:
                    Agent.learn()
            if done:
                done_times += 1
                LifeR.append(oneLife_totalR)
                oneLife_totalR = 0
                if done_times == lifes:
                    if plt_q:
                        Agent.statistical_values(sum(LifeR), q_change, action_change)
                    else:
                        Agent.statistical_values(sum(LifeR))
                    print('episode: ', i_episode, ' epsilon: ', round(Agent.epsilon, 2), 'Each_Life_reward:', LifeR, 'total_reward:', sum(LifeR))
                    break

            observation = observation_
            total_steps += 1


run_game(6, env, agent, True)
data_save4cnn("DQN", env_name, brain, agent)
