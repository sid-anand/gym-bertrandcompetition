import numpy as np
import random
import matplotlib.pyplot as plt

from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv

class Q_Learner():

    def __init__(self, env, num_agents=2, m=15, alpha=0.05, beta=0.2, gamma=0.99, epochs=50):

        self.env = env
        self.num_agents = num_agents
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epochs = epochs

        self.players = [ 'agent_' + str(i) for i in range(num_agents)]

    def train(self):

        q_table = [{}] * self.num_agents

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
                    if observation not in q_table[agent]:
                        q_table[agent][observation] = [0] * self.m

                    if random.uniform(0, 1) < epsilon:
                        actions_dict[self.players[agent]] = self.env.action_space.sample()
                    else:
                        actions_dict[self.players[agent]] = np.argmax(q_table[agent][observation])

                next_observation, reward, done, info = self.env.step(actions_dict)
                done = done['__all__']

                next_observation = str(next_observation)

                last_values = [0] * self.num_agents
                Q_maxes = [0] * self.num_agents
                for agent in range(self.num_agents):
                    if next_observation not in q_table[agent]:
                        q_table[agent][next_observation] = [0] * self.m
                
                    last_values[agent] = q_table[agent][observation][actions_dict[self.players[agent]]]
                    Q_maxes[agent] = np.max(q_table[agent][next_observation])
                
                    q_table[agent][observation][actions_dict[self.players[agent]]] = ((1 - self.alpha) * last_values[agent]) + (self.alpha * (reward[self.players[agent]] + self.gamma * Q_maxes[agent]))

                reward_list.append(reward[self.players[0]])

                observation = next_observation

                loop_count += 1
                
            mean_reward = np.mean(reward_list)

            if i % 1 == 0:
                print(f"Epochs: {i}, \tLoop Count: {loop_count}, \t Epsilon: {epsilon}, \tMean Reward: {mean_reward}")
            
            # all_epochs.append(epochs)
            for agent in range(self.num_agents):
                all_rewards[agent].append(mean_reward)

    def plot(self, last_n = 1000):

        x = np.arange(last_n)
        # print(env.action_history[players[0]][-num_actions:])
        # print(env.action_price_space.take(env.action_history[players[0]][-num_actions:]))
        for player in self.players:
            plt.plot(x, self.env.action_price_space.take(self.env.action_history[player][-last_n:]), alpha=0.75, label=player)
        plt.plot(x, np.repeat(self.env.pM, last_n), 'r--', label='Monopoly')
        plt.plot(x, np.repeat(self.env.pN, last_n), 'b--', label='Nash')
        plt.xlabel('Steps')
        plt.ylabel('Price')
        plt.savefig('./figures/' + self.env.trainer_choice + '_with_' + str(self.env.num_agents) + '_agents_k_' + str(self.env.k) + '_for_' + str(self.env.epochs * self.env.max_steps) + '_steps,last_steps_' + str(last_n))
        plt.clf()