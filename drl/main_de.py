import gym
import os
import mujoco_py
from scipy.optimize import differential_evolution
from agent import Agent
from train import Train
from play import Play

import statistics
import math
#import adv_sci
#from scipy.optimize import minimize
#import scipy.optimize as opt
ENV_NAME = "Ant"
TRAIN_FLAG = False
#TRAIN_FLAG = True
test_env = gym.make(ENV_NAME + "-v2")# 実行する課題を設定
n_states = test_env.observation_space.shape[0]# 課題の状態と行動の数を設定
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
n_actions = test_env.action_space.shape[0]# 行動を取得
n_iterations = 20000
#lr = 3e-4
lr = 3e-5
epochs = 10
#clip_range = 0.2
clip_range = 0.1
mini_batch_size = 256
T = 2048

def func(k):
	rewardlog=[]
	for j in range(100):#100
		rewardlog.append(player.evaluate(k))
	return statistics.mean(rewardlog)

if __name__ == "__main__":
    print(f"number of states:{n_states}\n"
          f"action bounds:{action_bounds}\n"
          f"number of actions:{n_actions}")

    if not os.path.exists(ENV_NAME):
        os.mkdir(ENV_NAME)
        os.mkdir(ENV_NAME + "/logs")

    env = gym.make(ENV_NAME + "-v2")
    env.seed(100)

    agent = Agent(n_states=n_states,
                  n_iter=n_iterations,
                  env_name=ENV_NAME,
                  action_bounds=action_bounds,
                  n_actions=n_actions,
                  lr=lr)
    if TRAIN_FLAG:
        trainer = Train(env=env,
                        test_env=test_env,
                        env_name=ENV_NAME,
                        agent=agent,
                        horizon=T,
                        n_iterations=n_iterations,
                        epochs=epochs,
                        mini_batch_size=mini_batch_size,
                        epsilon=clip_range)
        trainer.step()
        

    bounds = [(0.1,1.9),(0.1,1.9),(0.1,1.9),(0.1,1.9),(0.1,1.9),(0.1,1.9),(0.1,1.9),(0.1,1.9)]

    print("0.1-1.9")
    player = Play(env, agent, ENV_NAME)
    result = differential_evolution(func, bounds,maxiter=50,popsize=15,updating='immediate',disp=True)
    print(result)
