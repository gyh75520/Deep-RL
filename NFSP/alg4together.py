import player
import random
import sys
import time
import threading
from NFSP.Agent import Agent
from NFSP.Brain import Brain as brain

restore_flag = False

Brain = brain(
    n_actions=3,
    n_features=248,
    restore=restore_flag,
    checkpoint_dir='My_NFSP_two_play_Net',
    output_graph=True,
)
RL = Agent(
    brain=Brain,
    n_actions=3,
    information_state_shape=(248,),
    replace_target_iter=100,
)


def game4palyer(ply):
    episode = 0
    step = 0

    while True:
        Total_reward = 0.0
        RL.set_policy_sigma()
        obser, reward, done = ply.reset()
        #obser = np.array(obser)
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
            #obser_ = np.array(obser_)
            # print('\nobser__', obser_, obser_.shape)
            #print('\nreward', reward)
            #print('\nstep', step)

            if done:
                obser_ = None
                RL.store_memory(obser, action, reward, obser_)
                #print('\nreward', reward)
                if restore_flag is False and step > 50:
                    print(ply.playerName, 'step', step)
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
        print("player:", ply.playerName, 'now:', Total_reward, "episode:", episode)


def main():

    port = 48777
    logpath = "/Users/howard/Texas_open_source/project_acpc_server/matchName1.log"
    playerName = "Bob"
    ply = player.Player(playerName, port, logpath)

    port2 = 16177
    playerName2 = "Alice"
    ply2 = player.Player(playerName2, port2, logpath)

    t1 = threading.Thread(target=game4palyer, args=(ply,))
    t2 = threading.Thread(target=game4palyer, args=(ply2,))
    t1.start()
    t2.start()


if __name__ == '__main__':
    main()
