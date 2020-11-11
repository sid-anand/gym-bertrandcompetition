from gym import Env, logger
from gym.spaces import Discrete, Tuple, Box
from ray.rllib.env import MultiAgentEnv
from gym.utils import colorize, seeding
import sys
from contextlib import closing
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

# cd OneDrive/Documents/Research/gym-bertrandcompetition/gym_bertrandcompetition/envs

class BertrandCompetitionDiscreteEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents = 2, c_i = 1, a_minus_c_i = 1, a_0 = 0, mu = 0.25, delta = 0.95, m = 15, xi = 0.1, k = 1, max_steps=200, plot=True, epochs=10, convergence=5, trainer_choice='DQN'):

        super(BertrandCompetitionDiscreteEnv, self).__init__()
        self.num_agents = num_agents

        # Length of Memory
        self.k = k

        # Marginal Cost
        self.c_i = c_i

        # Number of Discrete Prices
        self.m = m

        # Product Quality Indexes
        self.a = np.array([c_i + a_minus_c_i])

        # Product Quality Index: Outside Good
        self.a_0 = a_0

        # Index of Horizontal Differentiation
        self.mu = mu

        # Nash Equilibrium Price
        self.pN = c_i

        # Monopoly Equilibrium Price
        monopoly_profit = []
        price_range = np.arange(0, 100, 0.1)
        for i in price_range:
            monopoly_profit.append((i - c_i) * self.demand(self.a, i, self.mu))
        self.pM = price_range[np.argmax(monopoly_profit)]

        # MultiAgentEnv Action Space
        self.action_space = Discrete(m)
        
        # MultiAgentEnv Observation Space
        if k > 0:
            self.numeric_low = np.array([0] * (k * num_agents))
            numeric_high = np.array([m] * (k * num_agents))
            self.observation_space = Box(self.numeric_low, numeric_high, dtype=int)
        else:
            self.numeric_low = np.array([0] * num_agents)
            numeric_high = np.array([m] * num_agents)
            self.observation_space = Box(self.numeric_low, numeric_high, dtype=int)

        self.action_price_space = np.linspace(self.pN - xi * (self.pM - self.pN), self.pM + xi * (self.pM - self.pN), m)
        self.reward_range = (-float('inf'), float('inf'))
        self.current_step = None
        self.max_steps = max_steps
        self.plot = plot
        self.epochs = epochs
        self.convergence = convergence
        self.trainer_choice = trainer_choice
        self.players = [ 'agent_' + str(i) for i in range(num_agents)]
        self.action_history = {}

        for i in range(num_agents):
            if self.players[i] not in self.action_history:
                # self.action_history[self.players[i]] = [0] * k
                self.action_history[self.players[i]] = []
                for _ in range(convergence):
                    self.action_history[self.players[i]].append(self.action_space.sample())

        self.reset()

    def demand(self, a, p, mu):
        ''' Demand as a function of product quality indexes, price, and mu. '''
        q = np.exp((a - p) / mu) / (np.sum(np.exp((a - p) / mu)) + np.exp(self.a_0 / mu))
        return q

    def step(self, actions_dict):
        ''' MultiAgentEnv Step'''

        actions_idx = np.array(list(actions_dict.values())).flatten()

        for i in range(actions_idx.size):
            self.action_history[self.players[i]].append(actions_idx[i])

        reward = np.array([0.0] * self.num_agents)

        if self.k > 0:
            obs_players = np.array([self.action_history[self.players[i]][-self.k:] for i in range(self.num_agents)]).flatten()
            observation = dict(zip(self.players, [obs_players for i in range(self.num_agents)]))
        else:
            observation = dict(zip(self.players, [self.numeric_low for _ in range(self.num_agents)]))

        actions = self.action_price_space.take(actions_idx)

        if np.max(actions) > self.c_i:
            min_price = min(actions[actions > self.c_i], default = self.c_i)
            total_profit = (min_price - self.c_i) * self.demand(self.a, min_price, self.mu)
            min_price_idxs = np.where(actions == min_price)[0]
            reward[min_price_idxs] = total_profit / min_price_idxs.size

        reward = dict(zip(self.players, reward))
        # done = {'__all__': self.current_step == self.max_steps}
        done = {'__all__': np.all(np.array(self.action_history[self.players[0]][-self.convergence:]) == self.action_history[self.players[0]][-1])}
        # done = {'__all__': True}
        info = dict(zip(self.players, [{}]*self.num_agents))

        # print('Obs:', self.obs_n)
        # print('Reward:', reward)
        # print('Done:', done)
        # print('Info:', info)

        # print('Actions:', actions)
        # print('Reward:', reward)

        self.current_step += 1

        n = len(self.action_history[self.players[0]])

        if self.plot and n == self.epochs * self.max_steps:
            x = np.arange(n)
            for player in self.players:
                plt.plot(x, self.action_price_space.take(self.action_history[player]), alpha=0.75, label=player)
            plt.plot(x, np.repeat(self.pM, n), 'r--', label='Monopoly')
            plt.plot(x, np.repeat(self.pN, n), 'b--', label='Nash')
            plt.xlabel('Steps')
            plt.ylabel('Price')
            plt.title(self.trainer_choice + ' with ' + str(self.num_agents) + ' agents and k=' + str(self.k) + ' for ' + str(self.epochs * self.max_steps) + ' Steps')
            plt.legend(loc='upper left')
            plt.savefig(self.trainer_choice + 'with' + str(self.num_agents) + 'agentsk' + str(self.k) + 'for' + str(self.epochs * self.max_steps) + 'Steps')
            plt.clf()

        return observation, reward, done, info

    def reset(self):
        self.current_step = 0

        # random_observation = np.random.randint(self.m, size=(self.k * self.num_agents))

        # Reset to action of 0
        # random_action = np.random.randint(1, size=self.num_agents)

        # Reset to random action
        random_action = np.random.randint(self.m, size=self.num_agents)

        for i in range(random_action.size):
            self.action_history[self.players[i]].append(random_action[i])

        if self.k > 0:
            obs_players = np.array([self.action_history[self.players[i]][-self.k:] for i in range(self.num_agents)]).flatten()
            observation = dict(zip(self.players, [obs_players for i in range(self.num_agents)]))
        else:
            observation = dict(zip(self.players, [self.numeric_low for _ in range(self.num_agents)]))
        # print(random_observation)

        # observation = {}
        # for player in self.players:
        #     observation[player] = random_observation
        #     # self.action_history[player] = self.action_history[player] + observation[player].tolist()
        #     self.action_history[player].append()
        #     # self.action_history[player].append(0)
        return observation

    def render(self, mode='human'):
        raise NotImplementedError



# Tests

# bcd = BertrandCompetitionDiscreteEnv()
# obs = bcd.reset()
# print(obs)
# print([0] * 0)

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