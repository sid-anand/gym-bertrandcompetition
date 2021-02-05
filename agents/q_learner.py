import numpy as np
import random
import matplotlib.pyplot as plt

from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv

class Q_Learner():

    def __init__(self, env, num_agents=2, m=15, alpha=0.05, beta=0.2, delta=0.99, supervisor=False, sessions=1, log_frequency=10000):

        self.env = env
        if supervisor:
            self.num_agents = num_agents + 1
            self.agents = [ 'agent_' + str(i) for i in range(num_agents)] + ['supervisor']
        else:
            self.num_agents = num_agents
            self.agents = [ 'agent_' + str(i) for i in range(num_agents)]
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.supervisor = supervisor
        self.sessions = sessions
        self.log_frequency = log_frequency

    def train(self):
        '''Train to fill q_table'''

        self.q_table = [{}] * self.num_agents

        # For plotting metrics
        all_rewards = [ [] for _ in range(self.num_agents) ] # store the penalties per episode

        for i in range(self.sessions):    

            observation = self.env.reset()
            # if self.supervisor:
            #     supervisor_observation = observation.pop('supervisor') # Try giving current actions as observation to supervisor!
            observation = str(observation)

            loop_count = 0
            reward_list = []
            done = False
            
            while not done:

                # epsilon = np.exp(-1 * self.beta * i)
                epsilon = np.exp(-1 * self.beta * loop_count)

                actions_dict = {}
                for agent in range(self.num_agents):
                    if observation not in self.q_table[agent]:
                        if self.agents[agent] == 'supervisor':
                            self.q_table[agent][observation] = [0] * (self.num_agents - 1)
                        else:
                            self.q_table[agent][observation] = [0] * self.m

                    if random.uniform(0, 1) < epsilon:
                        actions_dict[self.agents[agent]] = self.env.action_spaces[self.agents[agent]].sample()
                    else:
                        actions_dict[self.agents[agent]] = np.argmax(self.q_table[agent][observation])

                next_observation, reward, done, info = self.env.step(actions_dict)
                done = done['__all__']

                next_observation = str(next_observation)

                last_values = [0] * self.num_agents
                Q_maxes = [0] * self.num_agents
                for agent in range(self.num_agents):
                    if next_observation not in self.q_table[agent]:
                        if self.agents[agent] == 'supervisor':
                            self.q_table[agent][next_observation] = [0] * (self.num_agents - 1)
                        else:
                            self.q_table[agent][next_observation] = [0] * self.m

                    last_values[agent] = self.q_table[agent][observation][actions_dict[self.agents[agent]]]
                    Q_maxes[agent] = np.max(self.q_table[agent][next_observation])
                
                    self.q_table[agent][observation][actions_dict[self.agents[agent]]] = ((1 - self.alpha) * last_values[agent]) + (self.alpha * (reward[self.agents[agent]] + self.delta * Q_maxes[agent]))

                reward_list.append(reward[self.agents[0]])

                observation = next_observation

                loop_count += 1

                if loop_count % self.log_frequency == 0:
                    mean_reward = np.mean(reward_list[-self.log_frequency:])
                    print(f"Session: {i}, \tLoop Count: {loop_count}, \t Epsilon: {epsilon}, \tMean Reward: {mean_reward}")
                
            mean_reward = np.mean(reward_list)

            print(f"Session: {i}, \tLoop Count: {loop_count}, \t Epsilon: {epsilon}, \tMean Reward: {mean_reward}")
            
            for agent in range(self.num_agents):
                all_rewards[agent].append(mean_reward)

    def eval(self, observation, n=20):
        '''Eval q_table'''

        for i in range(n):

            observation = str(observation)

            actions_dict = {}
            for agent in range(self.num_agents):
                if observation not in self.q_table[agent]:
                    self.q_table[agent][observation] = [0] * self.m

                actions_dict[self.agents[agent]] = np.argmax(self.q_table[agent][observation])

            next_observation, reward, done, info = self.env.step(actions_dict)
            done = done['__all__']

            observation = str(next_observation)