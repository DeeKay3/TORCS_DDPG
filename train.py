import numpy as np
import torch
import gym
import random
from collections import namedtuple
from collections import defaultdict
from agent.ddpg import Ddpg
from agent.ActorNetwork import ActorNetwork
from agent.CriticNetwork import CriticNetwork
from agent.random_process import OrnsteinUhlenbeckProcess
from gym_torcs import TorcsEnv
import pickle
import json


def train(device):
    cont = False #Dont forget to change start sigma
    train_all = True #Set True if train all network, False if train only break part of the network(Dont forget to change clip_grad!)
    env = TorcsEnv(path="/usr/local/share/games/torcs/config/raceman/quickrace.xml")
    insize = env.observation_space.shape[0]
    outsize = env.action_space.shape[0]

    hyperparams = {
                "lrvalue": 0.001, #0.001
                "lrpolicy": 0.0001, #0.0001
                "gamma": 0.95,
                "episodes": 100000,
                "sigma_episode": 500,
                "buffersize": 300000,
                "tau": 0.001,
                "batchsize": 32,
                "start_sigma": 0.9, #0.9
                "end_sigma": 0.05,
                "theta": 0.15,
                "maxlength": 10000,
                "clipgrad": True
    }
    HyperParams = namedtuple("HyperParams", hyperparams.keys())
    hyprm = HyperParams(**hyperparams)

    datalog = defaultdict(list)
    
    valuenet = CriticNetwork(insize, outsize)
    policynet = ActorNetwork(insize)
    agent = Ddpg(valuenet, policynet, buffersize=hyprm.buffersize)
    agent.policynet.change_grad_all_except_brake(train_all)
    if train_all == False:
        agent.policynet.init_brake()
    agent.to(device)
    if cont:
        agent.load_state_dict(torch.load('best_agent_dict'))
        # agent.opt_policy.load_state_dict(torch.load('best_agent_policy_opt'))
        # agent.opt_value.load_state_dict(torch.load('best_agent_value_opt'))

    
    reward_list = []
    td_list = []
    best_reward = -np.inf

    for eps in range(hyprm.episodes):
        #print("Im here too")
        state = env.reset(relaunch= True, render=False, sampletrack=True)
        episode_reward = 0
        episode_value = 0
        sigma = (hyprm.start_sigma-hyprm.end_sigma)*(max(0, 1-eps/hyprm.sigma_episode)) + hyprm.end_sigma
        randomprocess = OrnsteinUhlenbeckProcess(hyprm.theta, sigma, outsize)
        for i in range(hyprm.maxlength):
            #print("Start iteration")
            torch_state = agent._totorch(state, torch.float32).view(1, -1)
            action, value = agent.act(torch_state)
            #print("Once: ", action[2])
            action =  action.to("cpu").squeeze() # + randomprocess.noise() 
            action.clamp_(-1, 1)
            #print("Sonra: ", action[2])
            #print("Once: ", action)
            #print("Sonra: ", action_new)
            #action[1] = (action[1]+1)/2
            action = np.concatenate([action[:2], [-1]])
            # if np.random.rand() < sigma:
            #     action[2] += np.random.rand()/2
            # if np.random.rand() > 0.1:
            #     action[2] = 0
            # action_step = action.numpy()
            # action_step[2] = 2*action_step[2] - 1
            # if np.random.rand() < sigma:
            #     action[2] *= -1

            next_state, reward, done, _ = env.step(action)
            #next_state, reward, done, _ = env.step(action_new)
            #print("Im here")
            agent.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.buffer) > hyprm.batchsize:
                # for params in agent.policynet.brake.parameters():
                #     print(params)
                value_loss, policy_loss = agent.update(hyprm.gamma, hyprm.batchsize, hyprm.tau, hyprm.lrvalue, hyprm.lrpolicy, hyprm.clipgrad)
                if random.uniform(0, 1) < 0.01:
                    td_list.append(value_loss)
                    datalog["td error"].append(value_loss)
                    datalog["avearge policy value"].append(policy_loss)

            if done:
                break
            state = next_state
        datalog["epsiode length"].append(i)
        datalog["total reward"].append(episode_reward)
        average_reward = torch.mean(torch.tensor(datalog["total reward"][-20:])).item()
        reward_list.append(average_reward)
        # with open("reward_list", "wb") as fp:
        #     pickle.dump(1, reward_list)
        # with open("td_list", "wb") as tp:
        #     pickle.dump(1, td_list)
        #Load
        '''
        with open("reward_list", "rb) as fp:
            list = pickle.load(fp)
        '''
        if eps%20 == 0:
            torch.save(agent.state_dict(), "models/agent_{}_dict".format(eps))
            torch.save(agent.opt_policy.state_dict(), "models/agent_{}_policy_opt".format(eps))
            torch.save(agent.opt_value.state_dict(), "models/agent_{}_value_opt".format(eps))

        if average_reward > best_reward:
            best_reward = average_reward
            torch.save(agent.state_dict(), "best_agent_dict")
            torch.save(agent.opt_policy.state_dict(), "best_agent_policy_opt")
            torch.save(agent.opt_value.state_dict(), "best_agent_value_opt")
        print("\r Processs percentage: {:2.1f}%, Average reward: {:2.3f}, Best reward: {:2.3f}".format(eps/hyprm.episodes*100, average_reward, best_reward), end="", flush=True)
        json.dump(datalog, open("plot_data.csv", 'w'))
    print("")



if __name__ == "__main__":
    train("cuda")