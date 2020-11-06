from gym import Env, logger
from gym.spaces import Discrete, Tuple, Dict, Box
from ray.rllib.env import MultiAgentEnv
from gym.utils import colorize, seeding
import sys
from contextlib import closing
import numpy as np
from io import StringIO
# from fixedlist import FixedList

# cd OneDrive/Documents/Research/gym-bertrandcompetition/gym_bertrandcompetition/envs

class BertrandCompetitionDiscreteEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents = 2, c_i = 1, a_0 = 0, mu = 0.25, delta = 0.95, m = 15, xi = 0.1, k = 1, max_steps=200):

        super(BertrandCompetitionDiscreteEnv, self).__init__()
        self.num_agents = num_agents
        self.k = k
        self.c_i = c_i
        self.a_0 = a_0
        self.mu = mu
        self.pN = c_i
        self.current_step = None
        self.max_steps = max_steps
        self.players = [ 'agent_' + str(i) for i in range(num_agents)]
        self.a = np.array([2.0])
        self.action_history = {}
        for i in range(num_agents):
            if self.players[i] not in self.action_history:
                self.action_history[self.players[i]] = [0]

        monopoly_profit = []
        price_range = np.arange(0, 100, 0.1)
        for i in price_range:
            monopoly_profit.append((i - c_i) * self.demand(self.a, i, self.mu))
        self.pM = price_range[np.argmax(monopoly_profit)]

        

        self.action_space = Discrete(m)
        self.numeric_low = np.array([0] * (k * num_agents))
        numeric_high = np.array([m] * (k * num_agents))
        self.observation_space = Box(self.numeric_low, numeric_high, dtype=int)

        self.action_price_space = np.linspace(self.pN - xi * (self.pM - self.pN), self.pM + xi * (self.pM - self.pN), m)

        self.reward_range = (-float('inf'), float('inf'))
        self.reset()

    def demand(self, a, p, mu):
        q = np.exp((a - p) / mu) / (np.sum(np.exp((a - p) / mu)) + np.exp(self.a_0 / mu))
        return q

    def step(self, actions_dict):

        actions_idx = np.array(list(actions_dict.values())).flatten()

        for i in range(actions_idx.size):
            self.action_history[self.players[i]].append(actions_idx[i])

        reward = np.array([0.0] * self.num_agents)

        obs_players = np.array([self.action_history[self.players[i]][-self.k:] for i in range(self.num_agents)]).flatten()

        observation = dict(zip(self.players, [obs_players for i in range(self.num_agents)]))

        actions = self.action_price_space.take(actions_idx)

        if np.max(actions) > self.c_i:
            min_price = min(actions[actions > self.c_i], default = self.c_i)
            total_profit = (min_price - self.c_i) * self.demand(self.a, min_price, self.mu) # self.demand(min_price)
            min_price_idxs = np.where(actions == min_price)[0]
            reward[min_price_idxs] = total_profit / min_price_idxs.size

        reward = dict(zip(self.players, reward))
        done = {'__all__': self.current_step == self.max_steps}
        info = dict(zip(self.players, [{}]*self.num_agents))
        # print('Obs:', self.obs_n)
        # print('Reward:', reward)
        # print('Done:', done)
        # print('Info:', info)

        # print('Actions:', actions)
        # print('Reward:', reward)

        self.current_step += 1

        return observation, reward, done, info

    def reset(self):
        self.current_step = 0
        observation = [self.numeric_low for _ in range(self.num_agents)]
        return dict(zip(self.players, observation))

    def render(self, mode='human'):
        raise NotImplementedError




# bcd = BertrandCompetitionDiscreteEnv()
# obs = bcd.reset()
# print(obs)

# for i in range(14):
#     print()
#     obs, rewards, dones, infos = bcd.step(actions_dict={'agent_0': i, 'agent_1': i})
#     print('Obs:', obs)
#     print('Reward:', rewards)
#     print('Done:', dones)
#     print('Info:', infos)
#     print()
#     obs, rewards, dones, infos = bcd.step(actions_dict={'agent_0': i, 'agent_1': i+1})
#     print('Obs:', obs)
#     print('Reward:', rewards)
#     print('Done:', dones)
#     print('Info:', infos)