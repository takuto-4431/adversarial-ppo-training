import gym
import os
import mujoco_py

#import adv
from agent import Agent
from train import Train
from play import Play
import statistics
import math
import numpy as np
ENV_NAME = "Ant"
#TRAIN_FLAG = True
TRAIN_FLAG = False
test_env = gym.make(ENV_NAME + "-v2")

n_states = test_env.observation_space.shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
n_actions = test_env.action_space.shape[0]

n_iterations = 100000
#lr = 3e-4
lr = 3e-5
epochs = 10
#clip_range = 0.2
clip_range = 0.1
#mini_batch_size = 64
mini_batch_size = 256
T = 2048 #horizon

def func(k,t):
	rewardlog=[]
	for j in range(t):#15
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

    player = Play(env, agent, ENV_NAME)
    xb = np.array([1,1,1,1,1,1,1,1],dtype='float64')
    x0 = np.array([0.7, 1.3, 1.3 ,1.3, 0.7, 1.3, 1.3, 1.3],dtype='float64')
    x1 = np.array([1.29971116 ,1.27077509, 0.95531106 ,1.23545059 ,1.28993255 ,1.09364566,1.0482986  ,1.01935747],dtype='float64')
    x5=np.array([1.4857283,1.46883394,0.98718199,1.08767977,1.44459113,1.3478122,1.32691483,1.44538197],dtype='float64')
    print(player.evaluate(xb))
    """
    print("dnn",func(x0,100))
    print("de",func(x1,100))
    print("base",func(xb,100))
    """
    """rewardlog=0
    [0.7, 1.3, 1.3 1.3, 0.7, 1.3, 1.3, 1.3]
    per =50
    for j in range(per):
        rewardlog+=player.evaluate()
    rewardlog=rewardlog/per
    print(per,rewardlog)"""

