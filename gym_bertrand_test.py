import gym
import gym_bertrandcompetition
from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv

import ray
import numpy as np
import random
from ray.tune.registry import register_env
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.agents.dqn import DQNTrainer
# from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.ppo import PPOTrainer
# from ray.rllib.agents.qmix import QMIXTrainer
from ray.rllib.agents.pg import PGTrainer
from ray.tune.logger import pretty_print

# cd OneDrive/Documents/Research/gym-bertrandcompetition

# CHANGE PARAMETERS FOR TESTING
# Parameters
num_agents = 2
k = 1
m = 15
max_steps = 500
epochs = 50
plot = True
# choose from QL, DQN, PPO, A3C
trainer_choice = 'QL'

env = BertrandCompetitionDiscreteEnv(num_agents=num_agents, k=k, m=m, max_steps=max_steps, plot=plot, epochs=epochs, trainer_choice=trainer_choice)

config = {
    'env_config': {
        'num_agents': num_agents,
    },
    'env': 'Bertrand',
    'num_workers': num_agents,
    # 'eager': True,
    # 'use_pytorch': False,
    'train_batch_size': 200,
    'rollout_fragment_length': 200,
    'lr': 0.001
}

if trainer_choice != 'QL':
    register_env('Bertrand', lambda env_config: env)
    ray.init(num_cpus=4)

    if trainer_choice == 'DQN':
        trainer = DQNTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'PPO':
        trainer = PPOTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'A3C':
        trainer = A3CTrainer(config = config, env = 'Bertrand')

    s = "Epoch {:3d} / Reward Min: {:6.2f} / Mean: {:6.2f} / Max: {:6.2f} / Steps {:6.2f}"

    for i in range(epochs):
        result = trainer.train()

        print(s.format(
        i + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"]))

    # print(pretty_print(result))
else:
    # Q-learning

    players = [ 'agent_' + str(i) for i in range(num_agents)]

    q_table = [{}] * num_agents

    # Hyperparameters
    alpha = 0.05
    beta = 0.2
    gamma = 0.99

    # For plotting metrics
    # all_epochs = [] #store the number of epochs per episode
    all_rewards = [ [] for _ in range(num_agents) ] #store the penalties per episode

    for i in range(epochs * max_steps):

        # Is this correct? Will it always be given this case? Need to code convergence.
        observation = env.reset()

        # epochs, total_reward = 0, 0
        reward_list = []
        done = False
        
        # Currently only runs once each time, need to make this run until convergence.
        while not done:

            epsilon = np.exp(-1 * beta * i)

            observation = str(observation)

            actions_dict = {}
            for agent in range(num_agents):
                if observation not in q_table[agent]:
                    q_table[agent][observation] = [0] * m

                if random.uniform(0, 1) < epsilon:
                    actions_dict[players[agent]] = env.action_space.sample()
                else:
                    actions_dict[players[agent]] = np.argmax(q_table[agent][observation])

            next_observation, reward, done, info = env.step(actions_dict)

            next_observation = str(next_observation)

            last_values = [0] * num_agents
            Q_maxes = [0] * num_agents
            for agent in range(num_agents):
                if next_observation not in q_table[agent]:
                    q_table[agent][next_observation] = [0] * m
            
                last_values[agent] = q_table[agent][observation][actions_dict[players[agent]]]
                Q_maxes[agent] = np.max(q_table[agent][next_observation])
            
                q_table[agent][observation][actions_dict[players[agent]]] = ((1 - alpha) * last_values[agent]) + (alpha * (reward[players[agent]] + gamma * Q_maxes[agent]))

            reward_list.append(reward[players[0]])

            observation = next_observation
            
        mean_reward = np.mean(reward_list)

        if i % 500 == 0:
            # clear_output(wait=True)
            print(f"Episode: {i}, \t Epsilon {epsilon}, \tMean Reward: {mean_reward}")
        
        # all_epochs.append(epochs)
        for agent in range(num_agents):
            all_rewards[agent].append(mean_reward)