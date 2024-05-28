"""
###################################################
######### Agent Interface for Evaluation ##########
###################################################

To hand in an agent for evaluation implement the functions "convert_obs", "convert_action" and the "Agent" class.
For a clarification on the purpose of the different functions please read the function documentations and see the
task description info sheet.

This file will be used during the evaluation of your code. For that reason make sure that you do not use any imports that
are not part of the evaluation environment. Also, please refrain from using this file for your training code. For that
create and use separate scripts.
"""


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
import numpy as np



def convert_obs(obs: np.ndarray):
    """
    Pre-computation steps to convert the observations received from the environment into the format that can be
    fed to your agent.

    E.g. if your agent is a nn.Module and works on tensors, you would need to convert the observation to a tensor here.

    :param obs: 456-dimensional numpy ndarray containing the current observations of vehicle state, sensed cones and middle line
    :return: the converted obs that can be handled by the agent
    """

    # remove unnecessary information
    indices_to_remove = [4, 5, 7, 8, 9, 10] # indices of position, rgb_arrays
    
    # remove the information about the cones
    for i in range(16, 416):
        indices_to_remove.append(i)

    # remove unecessary information and make tensor
    converted_obs = np.delete(obs, indices_to_remove)
    converted_obs = torch.tensor(converted_obs)

    return converted_obs


def convert_action(action):
    """
    any potentially needed computation steps to convert the actions provided by your agent
    into the format that can be fed into the environment.

    E.g. if your agent is a nn.Module and returns a tensor as action, you would need to convert the action tensor
    to a numpy array here.

    :param action: the action returned by your agent
    :return: the converted action that can be used as input to the environment's step function
    """


    # example
    converted_action = action.detach().cpu().numpy()
    return converted_action


# PPO Agent
class Agent(nn.Module):
    """
    The Agent Class

    To be able to evaluate your Code, the agent needs to implement the function "get_action". Feel free to implement
    additional functions but make sure to not change the signature of "get_action" and the name of the class.

    The querying process of your model looks schematically like that:
    OBS -> convert_obs -> get_action -> convert_action -> ACTION
    where OBS is the observation the environment provides and ACTION is the action that is fed into the environment.
    """    
    
    
    def __init__(self, env):
        super().__init__()
        
        action_shape = env.action_space.shape[0]
        obs_shape = convert_obs(env.reset()[0]).shape[0]
        
        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(np.prod(action_shape)))

    def get_value(self, x):
        return self.critic(x)
        
    def get_action(self, x):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.normal.Normal(action_mean, action_std)
        action = probs.sample()
        return action

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.normal.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(), probs.entropy().sum(), self.critic(x)
    

# DDPG Agent:
class DDPG_Agent(nn.Module):
    """
    The Agent Class

    To be able to evaluate your Code, the agent needs to implement the function "get_action". Feel free to implement
    additional functions but make sure to not change the signature of "get_action" and the name of the class.

    The querying process of your model looks schematically like that:
    OBS -> convert_obs -> get_action -> convert_action -> ACTION
    where OBS is the observation the environment provides and ACTION is the action that is fed into the environment.
    """


    def __init__(self, env):
        super().__init__()

        action_shape = env.action_space.shape
        obs_shape = convert_obs(env.reset()[0]).shape[0]

        self.fc1 = nn.Linear(obs_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(action_shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low)
                                         / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low)
                                        / 2.0, dtype=torch.float32)
        )
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias
        
    def get_action(self, obs):
        return self.forward(obs)

    
# RPO Agent:
class RPO_Agent(nn.Module):
    """
    The Agent Class

    To be able to evaluate your Code, the agent needs to implement the function "get_action". Feel free to implement
    additional functions but make sure to not change the signature of "get_action" and the name of the class.

    The querying process of your model looks schematically like that:
    OBS -> convert_obs -> get_action -> convert_action -> ACTION
    where OBS is the observation the environment provides and ACTION is the action that is fed into the environment.
    """
    
    
    def __init__(self, envs, rpo_alpha):        
        super().__init__()
        obs_shape = 331
        neur = 512
        
        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer
        
        self.rpo_alpha = rpo_alpha
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), neur)),
            nn.Tanh(),
            layer_init(nn.Linear(neur, neur)),
            nn.Tanh(),
            layer_init(nn.Linear(neur, 1), std=1.0),
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), neur)),
            nn.Tanh(),
            layer_init(nn.Linear(neur, neur)),
            nn.Tanh(),
            layer_init(nn.Linear(neur, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, device, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.normal.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  # new to RPO
            # sample again to add stochasticity to the policy
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
            action_mean = action_mean + z
            probs = torch.distributions.normal.Normal(action_mean, action_std)

        return action, probs.log_prob(action).sum(), probs.entropy().sum(), self.critic(x)

    def get_action(self, x, device):
        
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.normal.Normal(action_mean, action_std)
        action = probs.sample()
        
        # sample again to add stochasticity to the policy
        #z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha).to(device)
        #action_mean = action_mean + z
        #probs = torch.distributions.normal.Normal(action_mean, action_std)

        return action