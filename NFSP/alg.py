import player
import random
import sys
import time
from NFSP.Agent import Agent
from NFSP.Brain import Brain as brain
import numpy as np

restore_flag = False

# restore 和 MAX_EPSILON 一起调整 多少回合数save
Brain = brain(
    n_actions=3,
    n_features=8,
    restore=restore_flag,
    checkpoint_dir='My_NFSP_play1_Net',
)
RL = Agent(
    brain=Brain,
    n_actions=3,
    information_state_shape=(8,),
)


def main():
    test = True
    if test:
        port = 48777
        logpath = "/Users/howard/Texas_open_source/project_acpc_server/matchName1.log"
        playerName = "Bob"
    else:
        port = int(sys.argv[1])
        logpath = sys.argv[2]
        playerName = sys.argv[3]

    ply = player.Player(playerName, port, logpath)
    f = open('log.txt', 'w')

    error = 0
    episode = 0
    step = 0

    while True:
        Total_reward = 0.0
        RL.set_policy_sigma()
        obser, reward, done = ply.reset()
        obser = np.array(obser)
        if done:
            Total_reward += reward
            episode += 1
            continue
        # 如果先开一局，对方先发牌且对方马上弃牌，就会导致reset后马上结束

        if obser is None:
            print('obser is None')
            break
        while True:
            action = RL.choose_action(obser)
            #print('\naction', action)
            obser_, reward, done = ply.step(action)
            #print('\nobser_', obser_, type(obser_))
            obser_ = np.array(obser_)
            #print('\nobser__', obser_, obser_.shape)
            #print('\nreward', reward)
            #print('\nstep', step)

            if done:
                obser_ = None
                RL.store_memory(obser, action, reward, obser_)
                #print('\nreward', reward)
                if restore_flag is False and step > 50:
                    print('step', step)
                    RL.learn()
                Total_reward += reward
                step += 1
                episode += 1
                break

            RL.store_memory(obser, action, reward, obser_)

            if restore_flag is False and step > 50:
                RL.learn()

            obser = obser_
            step += 1
        print('now:', Total_reward, "episode:", episode)
        if restore_flag is False and episode == 999:
            Brain.save()  # 存储神经网络


if __name__ == '__main__':
    main()
