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
        self.state = FixedList(n = k)

    def demand(a, p, mu):
        q = np.exp((a - p) / mu) / (np.sum(np.exp((a - p) / mu)) + np.exp(a[0] / mu))
        return q

bcd = BetrandCompetitionDiscreteEnv()