from gym import Env, logger
from gym.spaces import Discrete, Tuple, Dict
from ray.rllib.env import MultiAgentEnv
from gym.utils import colorize, seeding
import sys
from contextlib import closing
import numpy as np
from io import StringIO
# from fixedlist import FixedList

# cd OneDrive/Documents/Research/gym-bertrandcompetition/gym_bertrandcompetition/envs

class FixedList(list):

    def __init__(self, n):
        self.n = n

    def add(self, item):
        list.insert(self, 0, item)
        if len(self) > self.n: del self[-1]

class BertrandCompetitionDiscreteEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents = 2, c_i = 1, a_0 = 0, mu = 0.25, delta = 0.95, m = 15, xi = 0.1, k = 1, pN = 1, pM = 10, max_steps=200):
        # change k back to 1

        super(BertrandCompetitionDiscreteEnv, self).__init__()
        self.num_agents = num_agents
        self.c_i = c_i
        self.a_0 = a_0
        self.current_step = None
        self.max_steps = max_steps
        self.players = [ 'agent_' + str(i) for i in range(num_agents)]
        # self.action_space = {0: np.linspace(pN - xi * (pM - pN), pM + xi * (pM - pN), m), 1: np.linspace(pN - xi * (pM - pN), pM + xi * (pM - pN), m)}
        # self.observation_space = np.linspace(pN - xi * (pM - pN), pM + xi * (pM - pN), m)

        self.action_space = Discrete(m)
        # self.observation_space = Tuple(tuple(Discrete(m) for _ in range(num_agents)))
        self.observation_space = Tuple(tuple(Discrete(m) for _ in range(num_agents)))

        self.reward_range = (-float('inf'), float('inf'))
        # self.observation_range = (-float('inf'), float('inf'))
        # self.obs_n = FixedList(n = k)
        # self.obs_n.add([0.0, 0.0])
        self.action_price_space = np.linspace(pN - xi * (pM - pN), pM + xi * (pM - pN), m)
        # self.obs_n = (a_space[0], a_space[0])
        # self.agent_dones = None
        self.reset()

    # def demand(self, a, p, mu):
    #     q = np.exp((a - p) / mu) / (np.sum(np.exp((a - p) / mu)) + np.exp(self.a_0 / mu))
    #     return q

    def demand(self, price):
        return np.max([-price + 12, 0])

    def step(self, actions_dict):

        actions_idx = np.array(list(actions_dict.values())).flatten()

        reward = np.array([0.0] * self.num_agents)

        observation = dict(zip(self.players, [tuple(actions_idx) for i in range(self.num_agents)]))

        actions = self.action_price_space.take(actions_idx)

        if np.max(actions) > self.c_i:
            min_price = min(actions[actions > self.c_i], default = self.c_i)
            total_profit = (min_price - self.c_i) * self.demand(min_price) # self.demand(11, min_price, 0.05)
            min_price_idxs = np.where(actions == min_price)[0]
            reward[min_price_idxs] = total_profit / min_price_idxs.size

        reward = dict(zip(self.players, reward))
        done = {'__all__': self.current_step == self.max_steps}
        info = dict(zip(self.players, [{}]*self.num_agents))
        # print('Obs:', self.obs_n)
        # print('Reward:', reward)
        # print('Done:', done)
        # print('Info:', info)

        print('Actions:', actions)
        print('Reward:', reward)

        self.current_step += 1

        return observation, reward, done, info

    # # get reward for a particular agent
    # def _get_reward(self, agent):
    #     if self.reward_callback is None:
    #         return 0.0
    #     return self.reward_callback(agent, self.world)

    def reset(self):
        self.current_step = 0
        # self.agent_dones = [False for _ in range(self.num_agents)]
        observation = [tuple(0 for _ in range(self.num_agents)) for _ in range(self.num_agents)]
        return dict(zip(self.players, observation))

    def render(self, mode='human'):
        raise NotImplementedError


# bcd = BertrandCompetitionDiscreteEnv()
# for i in range(14):
#     print()
#     bcd.step([bcd.action_space[i], bcd.action_space[i]])
#     print()
#     bcd.step([bcd.action_space[i], bcd.action_space[i+1]])
# #     # bcd.step([bcd.action_space[i+1], bcd.action_space[i]])

# pN = 1
# pM = 10
# xi = 0.1
# m = 15

# bcd = BertrandCompetitionDiscreteEnv()
# print(bcd.observation_space)
# a_space = np.linspace(pN - xi * (pM - pN), pM + xi * (pM - pN), m)

# for i in range(14):
#     print()
#     bcd.step([a_space[i], a_space[i]])
#     print()
#     bcd.step([a_space[i], a_space[i+1]])
# #     # bcd.step([bcd.action_space[i+1], bcd.action_space[i]])

# pN = 1
# pM = 10
# xi = 0.1
# m = 15

# bcd = BertrandCompetitionDiscreteEnv()
# obs = bcd.reset()
# print(obs)
# # print(bcd.observation_space)
# a_space = np.linspace(pN - xi * (pM - pN), pM + xi * (pM - pN), m)

# for i in range(14):
#     print()
#     obs, rewards, dones, infos = bcd.step(actions_dict={'agent_0': a_space[i], 'agent_1': a_space[i]})
#     print('Obs:', obs)
#     print('Reward:', rewards)
#     print('Done:', dones)
#     print('Info:', infos)
#     print()
#     obs, rewards, dones, infos = bcd.step(actions_dict={'agent_0': a_space[i], 'agent_1': a_space[i+1]})
#     print('Obs:', obs)
#     print('Reward:', rewards)
#     print('Done:', dones)
#     print('Info:', infos)
# #     # bcd.step([bcd.action_space[i+1], bcd.action_space[i]])