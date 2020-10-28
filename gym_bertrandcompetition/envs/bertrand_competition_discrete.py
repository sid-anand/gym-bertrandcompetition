from gym import Env, logger
from gym.spaces import Discrete, Tuple
from gym.utils import colorize, seeding
import sys
from contextlib import closing
import numpy as np
from io import StringIO
from fixedlist import FixedList

# cd OneDrive/Documents/Research/gym-bertrandcompetition/gym_bertrandcompetition/envs

class BertrandCompetitionDiscreteEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n = 2, c_i = 1, a_0 = 0, mu = 0.25, delta = 0.95, m = 15, xi = 0.1, k = 2, pN = 1, pM = 10):
        # change k back to 1

        self.n = n
        self.c_i = c_i
        self.a_0 = a_0
        self.action_space = np.linspace(pN - xi * (pM - pN), pM + xi * (pM - pN), m)
        self.observation_space = self.action_space.copy()
        self.reward_range = (-float('inf'), float('inf'))
        self.obs_n = FixedList(n = k)
        

    # def demand(self, a, p, mu):
    #     q = np.exp((a - p) / mu) / (np.sum(np.exp((a - p) / mu)) + np.exp(self.a_0 / mu))
    #     return q

    def demand(self, price):
        return np.max([-price + 12, 0])

    def step(self, action_n):

        action_n = np.array(action_n)

        reward_n = np.array([0.0] * self.n)
        done_n = [True] * self.n
        info_n = {'n': []}

        self.obs_n.add(action_n.tolist())

        if np.max(action_n) > self.c_i:
            min_price = min(action_n[action_n > self.c_i], default = self.c_i)
            total_profit = (min_price - self.c_i) * self.demand(min_price) # self.demand(11, min_price, 0.05)
            min_price_idxs = np.where(action_n == min_price)[0]
            reward_n[min_price_idxs] = total_profit / min_price_idxs.size

        print(self.obs_n, reward_n, done_n, info_n)

        return self.obs_n, reward_n, done_n, info_n

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError


bcd = BertrandCompetitionDiscreteEnv()
for i in range(14):
    print()
    bcd.step([bcd.action_space[i], bcd.action_space[i]])
    print()
    bcd.step([bcd.action_space[i], bcd.action_space[i+1]])
    # bcd.step([bcd.action_space[i+1], bcd.action_space[i]])