# -*- coding: UTF-8 -*-
'''
model = DQN,DDQN,PDQN,PDDQN,DQN_PER,DDQN_PER,DQN_InAday,DQN_PER_Ipm...
'''
# -----------ContolGame------------
# CartPole-v1,MountainCar-v0,Acrobot-v1,Pendulum-v0
from run_ContolGame import run_Game
run_Game('DDQN_PER_Ipm', 'Acrobot-v1', episodes=400)  # model,env,episodes


# -----------AtariGame-------------
# from run_AtariGame import run_Game
# run_Game('DQN_PER', 'Breakout', lifes=5, episodes=40001)  # model,env,lifes,episodes
