from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv

import ray
import numpy as np
import random
from ray.tune.registry import register_env
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer

from gianluca_playground import logger

# CHANGE PARAMETERS FOR TESTING
# Parameters
num_agents = 2
k = 1
m = 15
max_steps = 1000000
convergence = 10000
epochs = 50

# choose from QL, DQN, PPO, A3C
trainer_choice = 'QL'

def log_stats(env):
    relative_average_final_price = 0
    for i in range (num_agents):
        relative_average_final_price += ( (env.prices[i] - env.pN)/(env.pM - env.pN) )/num_agents

    logger.record_tabular("monopoly_price", env.pM)
    logger.record_tabular("Bertrand_price", env.pN)
    logger.record_tabular("relative_final_price",relative_average_final_price)
    for i in range(num_agents):
        logger.record_tabular("price_agent_"+str(i), env.prices[i])
    logger.dump_tabular()

if __name__ == '__main__':
    env = BertrandCompetitionDiscreteEnv(num_agents=num_agents, mu=0.01, k=k, m=m, max_steps=max_steps, plot=False, epochs=epochs, convergence=convergence, trainer_choice=trainer_choice)

    config = {
        'env_config': {'num_agents': num_agents,},
        'env': 'Bertrand',
        'num_workers': num_agents,
        'train_batch_size': 200,
        'rollout_fragment_length': 200,
        'lr': 0.001
    }

    # Logger
    logger.configure(experiment_name=f"Bertrand_competition_discrete_k_{k}_trainer_{trainer_choice}",)

    if trainer_choice != 'QL':
        register_env('Bertrand', lambda env_config: env)
        ray.init(num_cpus=4)

        for _ in range(epochs):
            if trainer_choice == 'DQN':
                trainer = DQNTrainer(config=config, env='Bertrand')
            elif trainer_choice == 'PPO':
                trainer = PPOTrainer(config=config, env='Bertrand')
            elif trainer_choice == 'A3C':
                trainer = A3CTrainer(config=config, env='Bertrand')

            result = trainer.train()
            log_stats(env) #TODO: this is not working because env does not store prices. Fix it or find better way to evaluate
    else:
        # Q-learning

        players = ['agent_' + str(i) for i in range(num_agents)]

        # Hyperparameters
        alpha = 0.05
        beta = 0.2
        gamma = 0.99

        # For plotting metrics
        # all_epochs = [] #store the number of epochs per episode
        all_rewards = [ [] for _ in range(num_agents) ] #store the penalties per episode

        # for i in range(epochs * max_steps):
        for _ in range(epochs):

            q_table = [{}] * num_agents
            observation = env.reset()

            # epochs, total_reward = 0, 0
            loop_count = 0
            reward_list = []
            done = False

            while not done:

                epsilon = np.exp(-1 * beta * loop_count)

                observation = str(observation)

                actions_dict = {}
                temp_str = ""
                for agent in range(num_agents):
                    if observation not in q_table[agent]:
                        q_table[agent][observation] = [0] * m

                    if random.uniform(0, 1) < epsilon:
                        actions_dict[players[agent]] = env.action_space.sample()
                    else:
                        actions_dict[players[agent]] = np.argmax(q_table[agent][observation])

                    temp_str = temp_str + str(actions_dict[players[agent]]) + " "

                next_observation, reward, done, info = env.step(actions_dict)
                done = done['__all__']
                next_observation = str(next_observation)

                last_values = [0] * num_agents
                Q_maxes = [0] * num_agents
                for agent in range(num_agents):
                    if next_observation not in q_table[agent]:
                        q_table[agent][next_observation] = [0] * m

                    last_values[agent] = q_table[agent][observation][actions_dict[players[agent]]]
                    Q_maxes[agent] = np.max(q_table[agent][next_observation])

                    q_table[agent][observation][actions_dict[players[agent]]] = ((1 - alpha) * last_values[agent]) + \
                                                                                (alpha * (reward[players[agent]] +
                                                                                          gamma * Q_maxes[agent]))

                reward_list.append(reward[players[0]])

                observation = next_observation

                loop_count += 1

            log_stats(env)