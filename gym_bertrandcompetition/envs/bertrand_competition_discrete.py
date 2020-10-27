from gym import Env, logger
from gym.spaces import Discrete, Tuple
from gym.utils import colorize, seeding
import sys
from contextlib import closing
import numpy as np
from io import StringIO
from fixedlist import FixedList

class BetrandCompetitionDiscreteEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n = 2, c_i = 1, a_0 = 0, mu = 0.25, delta = 0.95, m = 15, xi = 0.1, k = 1, pN = 1, pM = 10):

        self.action_space = np.linspace(pN - xi * (pM - pN), pM + xi * (pM - pN), m)
        self.observation_space = FixedList(n = k)
        self.reward_range = (-float('inf'), float('inf'))
        

    def demand(a, p, mu):
        q = np.exp((a - p) / mu) / (np.sum(np.exp((a - p) / mu)) + np.exp(a[0] / mu))
        return q

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}

        return obs_n, reward_n, done_n, info_n

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


bcd = BetrandCompetitionDiscreteEnv()