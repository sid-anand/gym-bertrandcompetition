from gym import Env, logger
from gym.spaces import Discrete, Tuple, Box
from ray.rllib.env import MultiAgentEnv
from gym.utils import colorize, seeding
import sys
from contextlib import closing
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy import optimize
import warnings

# warnings.filterwarnings('ignore')

class BertrandCompetitionDiscreteEnv(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents = 2, c_i = 1, a_minus_c_i = 1, a_0 = 0, mu = 0.25, delta = 0.95, m = 15, xi = 0.1, k = 1, max_steps=200, sessions=1, convergence=5, trainer_choice='DQN', supervisor=False, use_pickle=False, path=''):

        super(BertrandCompetitionDiscreteEnv, self).__init__()
        self.num_agents = num_agents

        # Length of Memory
        self.k = k

        # Marginal Cost
        self.c_i = c_i

        # Number of Discrete Prices
        self.m = m

        # Product Quality Indexes
        a = np.array([c_i + a_minus_c_i] * num_agents)
        self.a = a

        # Product Quality Index: Outside Good
        self.a_0 = a_0

        # Index of Horizontal Differentiation
        self.mu = mu

        # Nash Equilibrium Price
        def nash_func(p):
            ''' Derviative for demand function '''
            denominator = np.exp(a_0 / mu)
            for i in range(num_agents):
                denominator += np.exp((a[i] - p[i]) / mu)
            function_list = []
            for i in range(num_agents):
                term = np.exp((a[i] - p[i]) / mu)
                first_term = term / denominator
                second_term = (np.exp((2 * (a[i] - p[i])) / mu) * (-c_i + p[i])) / ((denominator ** 2) * mu)
                third_term = (term * (-c_i + p[i])) / (denominator * mu)
                function_list.append((p[i] - c_i) * (first_term + second_term - third_term))
            return function_list

        # Finding root of derivative for demand function
        nash_sol = optimize.root(nash_func, [2] * num_agents)
        self.pN = nash_sol.x[0]
        print('Nash Price:', self.pN)

        # # Finding Nash Price by iteration
        # # Make sure this tries all possibilities
        # price_range = np.arange(0, 2.5, 0.01)
        # nash_temp = 0
        # for i in price_range:
        #     p = [i] * num_agents
        #     first_agent_profit = (i - c_i) * self.demand(self.a, p, self.mu, 0)
        #     new_profit = []
        #     for j in price_range:
        #         p[0] = j
        #         new_profit.append((j - c_i) * self.demand(self.a, p, self.mu, 0))
        #     if first_agent_profit >= np.max(new_profit):
        #         nash_temp = i
        # self.pN = nash_temp
        # print('Nash Price:', self.pN)

        # Monopoly Equilibrium Price
        def monopoly_func(p):

            # Below is for finding each agent's monopoly price (currently unnecessary)
            # function_list = []
            # for i in range(num_agents):
            #     function_list.append(-(p[i] - c_i) * self.demand(self.a, p, self.mu, i))
            # return function_list
            
            return -(p[0] - c_i) * self.demand(self.a, p, self.mu, 0)

        monopoly_sol = optimize.minimize(monopoly_func, 0)
        self.pM = monopoly_sol.x[0]
        print('Monopoly Price:', self.pM)

        # # Finding Monopoly Price by iteration
        # # Make sure this tries all possibilities
        # price_range = np.arange(0, 2.5, 0.01)
        # monopoly_profit = []
        # for i in price_range:
        #     p = [i] * num_agents
        #     monopoly_profit.append((i - c_i) * self.demand(self.a, p, self.mu, 0) * num_agents)
        # self.pM = price_range[np.argmax(monopoly_profit)]
        # print('Monopoly Price:', self.pM)

        # MultiAgentEnv Action and Observation Space
        self.agents = ['agent_' + str(i) for i in range(num_agents)]
        self.observation_spaces = {}
        self.action_spaces = {}

        if k > 0:
            self.numeric_low = np.array([0] * (k * num_agents))
            numeric_high = np.array([m] * (k * num_agents))
            obs_space = Box(self.numeric_low, numeric_high, dtype=int)
        else:
            self.numeric_low = np.array([0] * num_agents)
            numeric_high = np.array([m] * num_agents)
            obs_space = Box(self.numeric_low, numeric_high, dtype=int)

        for agent in self.agents:
            self.observation_spaces[agent] = obs_space
            self.action_spaces[agent] = Discrete(m)

        if supervisor:
            self.observation_spaces['supervisor'] = obs_space
            self.action_spaces['supervisor'] = Discrete(num_agents)

        # MultiAgentEnv Action Space
        # self.action_space = Discrete(m)
        
        # MultiAgentEnv Observation Space
        # if k > 0:
        #     self.numeric_low = np.array([0] * (k * num_agents))
        #     numeric_high = np.array([m] * (k * num_agents))
        #     self.observation_space = Box(self.numeric_low, numeric_high, dtype=int)
        # else:
        #     self.numeric_low = np.array([0] * num_agents)
        #     numeric_high = np.array([m] * num_agents)
        #     self.observation_space = Box(self.numeric_low, numeric_high, dtype=int)

        self.action_price_space = np.linspace(self.pN - xi * (self.pM - self.pN), self.pM + xi * (self.pM - self.pN), m)
        self.reward_range = (-float('inf'), float('inf'))
        self.current_step = None
        self.max_steps = max_steps
        self.sessions = sessions
        self.convergence = convergence
        self.convergence_counter = 0
        self.trainer_choice = trainer_choice
        self.action_history = {}
        self.use_pickle = use_pickle
        self.supervisor = supervisor
        self.path = path

        if supervisor:
            self.savefile = 'discrete_' + trainer_choice + '_with_' + str(num_agents) + '_agents_k_' + str(k) + '_supervisor_' + supervisor + '_for_' + str(sessions) + '_sessions'
        else:
            self.savefile = 'discrete_' + trainer_choice + '_with_' + str(num_agents) + '_agents_k_' + str(k) + '_for_' + str(sessions) + '_sessions'

        for agent in self.agents:
            if agent not in self.action_history:
                self.action_history[agent] = [self.action_spaces[agent].sample()]

        if supervisor:
            self.action_history['supervisor'] = [self.action_spaces['supervisor'].sample()]

        self.reset()

    def demand(self, a, p, mu, agent_idx):
        ''' Demand as a function of product quality indexes, price, and mu. '''
        return np.exp((a[agent_idx] - p[agent_idx]) / mu) / (np.sum(np.exp((a - p) / mu)) + np.exp(self.a_0 / mu))

    def step(self, actions_dict):
        ''' MultiAgentEnv Step '''

        actions_idx = np.array(list(actions_dict.values())).flatten()

        # actions_idx = np.array([np.min(actions_idx)] * 2)
        # self.prices = self.action_price_space.take(actions_idx)
        # demand = [self.demand(self.a, self.prices, self.mu, 0), self.demand(self.a, self.prices, self.mu, 1)]
        # actions_idx[0] = 0
        # print(actions_idx)
        # temp_actions_idx = [self.m - 1,self.m - 1]
        # temp_actions_idx = [0, 0]
        # temp_actions_idx[np.argmin(actions_idx)] = np.max(actions_idx)
        # temp_actions_idx[np.argmax(actions_idx)] = np.min(actions_idx)
        # actions_idx = np.array(temp_actions_idx)
        # print(actions_idx)

        if self.use_pickle:
            with open(self.path + './arrays/' + self.savefile + '.pkl', 'ab') as f:
                pickle.dump(actions_idx, f)

        for i in range(self.num_agents):
            self.action_history[self.agents[i]].append(actions_idx[i])

        if self.supervisor:
            self.action_history['supervisor'].append(actions_idx[-1])

        if self.k > 0:
            obs_agents = np.array([self.action_history[self.agents[i]][-self.k:] for i in range(self.num_agents)], dtype=object).flatten()
            observation = dict(zip(self.agents, [obs_agents for i in range(self.num_agents)]))
        else:
            observation = dict(zip(self.agents, [self.numeric_low for _ in range(self.num_agents)]))

        self.prices = self.action_price_space.take(actions_idx[:self.num_agents])
        reward = np.array([0.0] * self.num_agents)

        if self.supervisor:
            total_demand = 0
            proportion = 3/4
            for i in range(self.num_agents):
                total_demand += self.demand(self.a, self.prices, self.mu, i)
            for i in range(self.num_agents):
                if i == actions_idx[-1]:
                    demand = total_demand * proportion
                else:
                    demand = total_demand * (1 - proportion)
                reward[i] = (self.prices[i] - self.c_i) * demand
        else:
            for i in range(self.num_agents):
                reward[i] = (self.prices[i] - self.c_i) * self.demand(self.a, self.prices, self.mu, i)

        reward = dict(zip(self.agents, reward))

        if self.action_history[self.agents[0]][-2] == self.action_history[self.agents[0]][-1]:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0

        if self.convergence_counter == self.convergence or self.current_step == self.max_steps:
            done = {'__all__': True}
        else:
            done = {'__all__': False}

        info = dict(zip(self.agents, [{}]*self.num_agents))

        if self.supervisor:
            if self.k > 0: 
                observation['supervisor'] = obs_agents
            else:
                observation['supervisor'] = self.numeric_low
            reward['supervisor'] = -np.sum(self.prices)
            info['supervisor'] = {}

        self.current_step += 1

        # print(observation, reward, done, info)

        return observation, reward, done, info

    def deviate(self, direction='down'):
        deviate_actions_dict = {}

        if direction == 'down':
            # First agent deviates to lowest price
            deviate_actions_dict[self.agents[0]] = 0
        elif direction == 'up':
            # First agent deviates to highest price
            deviate_actions_dict[self.agents[0]] = self.m - 1

        for agent in range(1, self.num_agents):
            # All other agents remain at previous price (large assumption)
            deviate_actions_dict[self.agents[agent]] = self.action_history[self.agents[agent]][-1]

        if self.supervisor:
            deviate_actions_dict['supervisor'] = self.action_history['supervisor'][-1]

        observation, _, _, _ = self.step(deviate_actions_dict)

        return observation

    def reset(self):
        self.current_step = 0

        # Reset to random action
        random_action = np.random.randint(self.m, size=self.num_agents)

        for i in range(random_action.size):
            self.action_history[self.agents[i]].append(random_action[i])

        if self.k > 0:
            obs_agents = np.array([self.action_history[self.agents[i]][-self.k:] for i in range(self.num_agents)], dtype=object).flatten()
            observation = dict(zip(self.agents, [obs_agents for i in range(self.num_agents)]))
        else:
            observation = dict(zip(self.agents, [self.numeric_low for _ in range(self.num_agents)]))

        if self.supervisor:
            if self.k > 0: 
                observation['supervisor'] = obs_agents
            else:
                observation['supervisor'] = self.numeric_low
            
        return observation

    def plot(self, window=1000, overwrite_id=0):
        '''Plot action history.'''
        warnings.filterwarnings('ignore')
        n = len(self.action_history[self.agents[0]])
        x = np.arange(n)
        for agent in self.agents:
            plt.plot(x, self.action_price_space.take(self.action_history[agent]), alpha=0.75, label=agent)
        for agent in self.agents:
            plt.plot(x, pd.Series(self.action_price_space.take(self.action_history[agent])).rolling(window=window).mean(), alpha=0.5, label=agent + ' MA')
        plt.plot(x, np.repeat(self.pM, n), 'r--', label='Monopoly')
        plt.plot(x, np.repeat(self.pN, n), 'b--', label='Nash')
        plt.xlabel('Steps')
        plt.ylabel('Price')
        plt.title(self.trainer_choice + ' with ' + str(self.num_agents) + ' agents and k=' + str(self.k) + ' Supervisor ' + self.supervisor + ' for ' + str(self.sessions) + ' Sessions')
        plt.legend(loc='upper left')
        plt.savefig('./figures/' + self.savefile + '_' + str(overwrite_id))
        plt.clf()

    def plot_last(self, last_n=1000, title_str = '', overwrite_id=0):
        x = np.arange(last_n)
        for agent in self.agents:
            plt.plot(x, self.action_price_space.take(self.action_history[agent][-last_n:]), alpha=0.75, label=agent)
        plt.plot(x, np.repeat(self.pM, last_n), 'r--', label='Monopoly')
        plt.plot(x, np.repeat(self.pN, last_n), 'b--', label='Nash')
        plt.xlabel('Steps')
        plt.ylabel('Price')
        plt.title(self.trainer_choice + ' with ' + str(self.num_agents) + ' agents and k=' + str(self.k) + ' Supervisor ' + self.supervisor + ' for ' + str(self.sessions) + ' Sessions, Last Steps ' + str(last_n) + title_str)
        plt.legend()
        plt.savefig('./figures/' + self.savefile + title_str + '_last_steps_' + str(last_n) + '_' + str(overwrite_id))
        plt.clf()

    def render(self, mode='human'):
        raise NotImplementedError