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

    def __init__(
            self, 
            num_agents = 2, 
            c = [1, 1], 
            a_minus_c = [1, 1], 
            a_0 = 0, 
            mu = 0.25, 
            delta = 0.95, 
            m = 15, 
            xi = 0.1, 
            k = 1, 
            max_steps=200, 
            sessions=1, 
            convergence=5, 
            trainer_choice='DQN', 
            supervisor=False, 
            proportion_boost=1.0, 
            use_pickle=False, 
            path='', 
            savefile=''
        ):

        super(BertrandCompetitionDiscreteEnv, self).__init__()
        self.num_agents = num_agents

        # Length of Memory
        self.k = k

        # Marginal Cost
        self.c = c

        # Number of Discrete Prices
        self.m = m

        # Product Quality Indexes
        a = np.array(c) + np.array(a_minus_c)
        self.a = a

        # Product Quality Index: Outside Good
        self.a_0 = a_0

        # Index of Horizontal Differentiation
        self.mu = mu

        ##############################################################

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
                second_term = (np.exp((2 * (a[i] - p[i])) / mu) * (-c[i] + p[i])) / ((denominator ** 2) * mu)
                third_term = (term * (-c[i] + p[i])) / (denominator * mu)
                function_list.append((p[i] - c[i]) * (first_term + second_term - third_term))
            return function_list

        # Finding root of derivative for demand function
        nash_sol = optimize.root(nash_func, [2] * num_agents)
        self.pN = nash_sol.x[0]
        print('Nash Price (for Agent 0):', self.pN)

        # # Finding Nash Price by iteration
        # # Make sure this tries all possibilities
        # price_range = np.arange(0, 2.5, 0.01)
        # nash_temp = 0
        # for i in price_range:
        #     p = [i] * num_agents
        #     first_player_profit = (i - c_i) * self.demand(self.a, p, self.mu, 0)
        #     new_profit = []
        #     for j in price_range:
        #         p[0] = j
        #         new_profit.append((j - c_i) * self.demand(self.a, p, self.mu, 0))
        #     if first_player_profit >= np.max(new_profit):
        #         nash_temp = i
        # self.pN = nash_temp
        # print('Nash Price:', self.pN)

        ############################################################################

        # Monopoly Equilibrium Price
        def monopoly_func(p):
            return -(p[0] - c[0]) * self.demand(self.a, p, self.mu, 0)

        monopoly_sol = optimize.minimize(monopoly_func, 0)
        self.pM = monopoly_sol.x[0]
        print('Monopoly Price (for Agent 0):', self.pM)

        # # Finding Monopoly Price by iteration
        # # Make sure this tries all possibilities
        # price_range = np.arange(0, 2.5, 0.01)
        # monopoly_profit = []
        # for i in price_range:
        #     p = [i] * num_agents
        #     monopoly_profit.append((i - c_i) * self.demand(self.a, p, self.mu, 0) * num_agents)
        # self.pM = price_range[np.argmax(monopoly_profit)]
        # print('Monopoly Price:', self.pM)

        ###############################################################################

        # Nash and Monopoly Profit

        # nash_profit = (self.pN - self.c_i) * self.demand(self.a, [self.pN, self.pN], self.mu, 0)
        # monopoly_profit = (self.pM - self.c_i) * self.demand(self.a, [self.pM, self.pM], self.mu, 0)

        # print('Nash Profit:', nash_profit)
        # print('Monopoly Profit:', monopoly_profit)

        ###############################################################################

        # Profit Gain Plot

        # val = []
        # # x_range = np.linspace(self.pN-0.05, self.pM+0.05, 100)
        # x_range = np.arange(1.45, 2.00, 0.05)

        # for i in x_range:
        #     profit = (i - self.c_i) * self.demand(self.a, [i, i], self.mu, 0)
        #     profit_gain = (profit - nash_profit) / (monopoly_profit - nash_profit)
        #     val.append(profit_gain)

        #     print(i, profit, profit_gain)

        # plt.plot(x_range, val, c='k')
        # plt.axvline(x=self.pN, c='b', ls='--', label='Nash Price')
        # plt.axvline(x=self.pM, c='r', ls='--', label='Monopoly Price')
        # plt.xlabel('Price')
        # plt.ylabel('Profit Gain Δ')
        # plt.legend()
        # plt.savefig('profit_gain')

        ################################################################################

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
        self.proportion_boost = proportion_boost
        self.path = path
        self.savefile = savefile

        # self.price_error = [[],[]]

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

        # Downward Price Step, below Nash
        # temp_actions_idx = [0, 0]
        # temp_actions_idx[np.argmax(actions_idx)] = np.min(actions_idx)
        # temp_actions_idx[1 - np.argmax(actions_idx)] = 0
        # actions_idx = np.array(temp_actions_idx)

        # Upward Price Step, above Monopoly
        # temp_actions_idx = [0, 0]
        # temp_actions_idx[np.argmax(actions_idx)] = self.m - 1
        # temp_actions_idx[1 - np.argmax(actions_idx)] = np.max(actions_idx)
        # actions_idx = np.array(temp_actions_idx)

        # Constant Decrease
        # temp_actions_idx = [0, 0]
        # temp_actions_idx[np.argmax(actions_idx)] = np.min(actions_idx)
        # temp_actions_idx[1 - np.argmax(actions_idx)] = np.clip(np.min(actions_idx) - 2, 0, self.m - 1)
        # self.price_error[0].append(np.abs(self.action_price_space.take(actions_idx[0]) - self.action_price_space.take(temp_actions_idx[0])))
        # self.price_error[1].append(np.abs(self.action_price_space.take(actions_idx[1]) - self.action_price_space.take(temp_actions_idx[1])))
        # actions_idx = np.array(temp_actions_idx)

        # Downward Price Step with Original Demand, below Nash
        # self.prices = self.action_price_space.take(actions_idx)
        # demand = [self.demand(self.a, self.prices, self.mu, 0), self.demand(self.a, self.prices, self.mu, 1)]
        # temp_actions_idx = [0, 0]
        # temp_actions_idx[np.argmax(actions_idx)] = np.min(actions_idx)
        # temp_actions_idx[1 - np.argmax(actions_idx)] = 0
        # actions_idx = np.array(temp_actions_idx)

        # Constant Decrease with Original Demand
        # self.prices = self.action_price_space.take(actions_idx)
        # demand = [self.demand(self.a, self.prices, self.mu, 0), self.demand(self.a, self.prices, self.mu, 1)]
        # temp_actions_idx = [0, 0]
        # temp_actions_idx[np.argmax(actions_idx)] = np.min(actions_idx)
        # temp_actions_idx[1 - np.argmax(actions_idx)] = np.clip(np.min(actions_idx) - 2, 0, self.m - 1)
        # self.price_error[0].append(np.abs(self.action_price_space.take(actions_idx[0]) - self.action_price_space.take(temp_actions_idx[0])))
        # self.price_error[1].append(np.abs(self.action_price_space.take(actions_idx[1]) - self.action_price_space.take(temp_actions_idx[1])))
        # actions_idx = np.array(temp_actions_idx)

        # Downward Price Step, at Nash
        # temp_actions_idx = [0, 0]
        # temp_actions_idx[np.argmax(actions_idx)] = np.min(actions_idx)
        # temp_actions_idx[1 - np.argmax(actions_idx)] = 1
        # self.price_error[0].append(np.abs(self.action_price_space.take(actions_idx[0]) - self.action_price_space.take(temp_actions_idx[0])))
        # self.price_error[1].append(np.abs(self.action_price_space.take(actions_idx[1]) - self.action_price_space.take(temp_actions_idx[1])))
        # actions_idx = np.array(temp_actions_idx)

        # Upward Price Step with Original Demand
        # self.prices = self.action_price_space.take(actions_idx)
        # demand = [self.demand(self.a, self.prices, self.mu, 0), self.demand(self.a, self.prices, self.mu, 1)]
        # temp_actions_idx = [0, 0]
        # temp_actions_idx[np.argmax(actions_idx)] = self.m - 1
        # temp_actions_idx[1 - np.argmax(actions_idx)] = np.max(actions_idx)
        # actions_idx = np.array(temp_actions_idx)

        # Fractional Decrease, Half
        # temp_actions_idx = [0, 0]
        # temp_actions_idx[np.argmax(actions_idx)] = np.min(actions_idx)
        # temp_actions_idx[1 - np.argmax(actions_idx)] = np.min(actions_idx) // 2
        # self.price_error[0].append(np.abs(self.action_price_space.take(actions_idx[0]) - self.action_price_space.take(temp_actions_idx[0])))
        # self.price_error[1].append(np.abs(self.action_price_space.take(actions_idx[1]) - self.action_price_space.take(temp_actions_idx[1])))
        # actions_idx = np.array(temp_actions_idx)

        # Fractional Decrease, Third
        # temp_actions_idx = [0, 0]
        # temp_actions_idx[np.argmax(actions_idx)] = np.min(actions_idx)
        # temp_actions_idx[1 - np.argmax(actions_idx)] = np.min(actions_idx) // 3
        # self.price_error[0].append(np.abs(self.action_price_space.take(actions_idx[0]) - self.action_price_space.take(temp_actions_idx[0])))
        # self.price_error[1].append(np.abs(self.action_price_space.take(actions_idx[1]) - self.action_price_space.take(temp_actions_idx[1])))
        # actions_idx = np.array(temp_actions_idx)

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
            # total_demand = 0
            # for i in range(self.num_agents):
            #     total_demand += self.demand(self.a, self.prices, self.mu, i)
            # demand_change = [0] * self.num_agents
            for i in range(self.num_agents):
                if i == actions_idx[-1]:
                    reward[i] = (self.prices[i] - self.c[i]) * (self.demand(self.a, self.prices, self.mu, i) * self.proportion_boost)
                    # demand_proportion = (self.demand(self.a, self.prices, self.mu, i) / total_demand) + self.proportion_boost
                    # demand_change[i] = np.abs(self.demand(self.a, self.prices, self.mu, i) - (self.demand(self.a, self.prices, self.mu, i) * self.proportion_boost))
                else:
                    reward[i] = (self.prices[i] - self.c[i]) * (self.demand(self.a, self.prices, self.mu, i) * ((2 - self.proportion_boost) / (self.num_agents - 1)))
                    # demand_proportion = (self.demand(self.a, self.prices, self.mu, i) / total_demand) - self.proportion_boost
                    # demand_change[i] = np.abs(self.demand(self.a, self.prices, self.mu, i) - (self.demand(self.a, self.prices, self.mu, i) * self.proportion_boost))
                # reward[i] = (self.prices[i] - self.c[i]) * (total_demand * demand_proportion)

            # if self.use_pickle:
            #     with open(self.path + './arrays/' + self.savefile + 'demand.pkl', 'ab') as f:
            #         pickle.dump(demand_change, f)
        else:
            for i in range(self.num_agents):
                reward[i] = (self.prices[i] - self.c[i]) * self.demand(self.a, self.prices, self.mu, i)
                # reward[i] = (self.prices[i] - self.c[i]) * demand[i]

        reward = dict(zip(self.agents, reward))

        if self.action_history[self.agents[0]][-2] == self.action_history[self.agents[0]][-1]:
            self.convergence_counter += 1
        else:
            self.convergence_counter = 0

        if self.convergence_counter == self.convergence or self.current_step == self.max_steps:
            done = {'__all__': True}
        else:
            done = {'__all__': False}

        info = dict(zip(self.agents, [{} for _ in range(self.num_agents)]))

        if self.supervisor:
            if self.k > 0: 
                observation['supervisor'] = obs_agents
            else:
                observation['supervisor'] = self.numeric_low
            reward['supervisor'] = -np.prod(self.prices)
            info['supervisor'] = {}

        self.current_step += 1

        return observation, reward, done, info

    def one_step(self):
        step_actions_dict = {}

        for agent in self.agents:
            step_actions_dict[agent] = self.action_history[agent][-1]

        if self.supervisor:
            step_actions_dict['supervisor'] = self.action_history['supervisor'][-1]

        observation, _, _, _ = self.step(step_actions_dict)

        return observation

    def deviate(self, direction='down'):
        deviate_actions_dict = {}

        if direction == 'down':
            # First agent deviates to lowest price
            deviate_actions_dict[self.agents[0]] = 3
        elif direction == 'up':
            # First agent deviates to highest price
            deviate_actions_dict[self.agents[0]] = self.m - 3

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
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        # plt.title(self.savefile.replace('_', ' ').title())
        plt.legend(loc='upper left')
        plt.savefig('./figures/' + self.savefile + '_' + str(overwrite_id))
        plt.clf()

    def plot_last(self, last_n=1000, window=None, title_str = '', overwrite_id=0):
        '''Plot action history.'''
        x = np.arange(last_n)
        for agent in self.agents:
            plt.plot(x, self.action_price_space.take(self.action_history[agent][-last_n:]), alpha=0.75, label=agent)
        if window is not None:
            for agent in self.agents:
                plt.plot(x, pd.Series(self.action_price_space.take(self.action_history[agent][-last_n:])).rolling(window=window).mean(), alpha=0.5, label=agent + ' MA')
        plt.plot(x, np.repeat(self.pM, last_n), 'r--', label='Monopoly')
        plt.plot(x, np.repeat(self.pN, last_n), 'b--', label='Nash')
        plt.xlabel('Steps')
        plt.ylabel('Price')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        # plt.title((self.savefile + title_str + ' Eval ' + str(last_n) ).replace('_', ' ').title())
        plt.legend()
        plt.savefig('./figures/' + self.savefile + title_str + '_eval_' + str(last_n) + '_' + str(overwrite_id))
        plt.clf()

    def render(self, mode='human'):
        raise NotImplementedError