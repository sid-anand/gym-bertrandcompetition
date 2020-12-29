import numpy as np
import random
import matplotlib.pyplot as plt

from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv

class Q_Learner():

    def __init__(self, env, num_agents=2, m=15, alpha=0.05, beta=0.2, delta=0.99, epochs=50):

        self.env = env
        self.num_agents = num_agents
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.epochs = epochs

        self.players = [ 'agent_' + str(i) for i in range(num_agents)]

    def train(self):
        '''Train to fill q_table'''

        self.q_table = [{}] * self.num_agents

        # For plotting metrics
        # all_epochs = [] #store the number of epochs per episode
        all_rewards = [ [] for _ in range(self.num_agents) ] # store the penalties per episode

        for i in range(self.epochs):    

            observation = self.env.reset()

            # epochs, total_reward = 0, 0
            loop_count = 0
            reward_list = []
            done = False
            
            while not done:

                epsilon = np.exp(-1 * self.beta * i)

                observation = str(observation)

                actions_dict = {}
                for agent in range(self.num_agents):
                    if observation not in self.q_table[agent]:
                        self.q_table[agent][observation] = [0] * self.m

                    if random.uniform(0, 1) < epsilon:
                        actions_dict[self.players[agent]] = self.env.action_space.sample()
                    else:
                        actions_dict[self.players[agent]] = np.argmax(self.q_table[agent][observation])

                next_observation, reward, done, info = self.env.step(actions_dict)
                done = done['__all__']

                next_observation = str(next_observation)

                last_values = [0] * self.num_agents
                Q_maxes = [0] * self.num_agents
                for agent in range(self.num_agents):
                    if next_observation not in self.q_table[agent]:
                        self.q_table[agent][next_observation] = [0] * self.m
                
                    last_values[agent] = self.q_table[agent][observation][actions_dict[self.players[agent]]]
                    Q_maxes[agent] = np.max(self.q_table[agent][next_observation])
                
                    self.q_table[agent][observation][actions_dict[self.players[agent]]] = ((1 - self.alpha) * last_values[agent]) + (self.alpha * (reward[self.players[agent]] + self.delta * Q_maxes[agent]))

                reward_list.append(reward[self.players[0]])

                observation = next_observation

                loop_count += 1
                
            mean_reward = np.mean(reward_list)

            if i % 1 == 0:
                print(f"Epochs: {i}, \tLoop Count: {loop_count}, \t Epsilon: {epsilon}, \tMean Reward: {mean_reward}")
            
            # all_epochs.append(epochs)
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

                actions_dict[self.players[agent]] = np.argmax(self.q_table[agent][observation])

            next_observation, reward, done, info = self.env.step(actions_dict)
            done = done['__all__']

            observation = str(next_observation)