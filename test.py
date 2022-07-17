import gym_torcs as gymt
import numpy as np
from my_agent.ddpg import Ddpg
from my_agent.ActorNetwork import ActorNetwork
from my_agent.CriticNetwork import CriticNetwork
import torch

env = gymt.TorcsEnv(path="/usr/local/share/games/torcs/config/raceman/quickrace.xml")

insize = env.observation_space.shape[0]
outsize = env.action_space.shape[0]


valuenet = CriticNetwork(insize, outsize)
policynet = ActorNetwork(insize)
agent = Ddpg(valuenet, policynet, buffersize=1)
agent.load_state_dict(torch.load('best_agent'))
#agent.to(device)

obs, reward, done, info = env.reset(relaunch=True, render=True)

while True:
    
    obs, reward, done, info = env.step()

    if (done):
        print("DONE")
        break
