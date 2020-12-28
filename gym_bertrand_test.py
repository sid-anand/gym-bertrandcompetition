import gym
import gym_bertrandcompetition
from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv
from gym_bertrandcompetition.envs.bertrand_competition_continuous import BertrandCompetitionContinuousEnv
from agents.q_learner import Q_Learner

import os
import pickle
import ray
import numpy as np
import matplotlib.pyplot as plt
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
max_steps = 1000
convergence = 5
epochs = 3
state_space = 'continuous' # 'discrete' or 'continuous'
# choose from QL, DQN, PPO, A3C
trainer_choice = 'A3C'

if state_space == 'discrete':
    env = BertrandCompetitionDiscreteEnv(num_agents=num_agents, k=k, m=m, max_steps=max_steps, epochs=epochs, convergence=convergence, trainer_choice=trainer_choice)
elif state_space == 'continuous':
    env = BertrandCompetitionContinuousEnv(num_agents=num_agents, k=k, max_steps=max_steps, epochs=epochs, trainer_choice=trainer_choice)

config = {
    'env_config': {
        'num_agents': num_agents,
    },
    'env': 'Bertrand',
    'num_workers': num_agents,
    'train_batch_size': 200,
    'rollout_fragment_length': 200,
    'explore': True, # Change this to False to evaluate
    'monitor': True,
    'log_level': 'WARN', # INFO for more
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

    savefile = './arrays/' + state_space + '_' + trainer_choice + '_with_' + str(num_agents) + '_agents_k_' + str(k) + '_for_' + str(epochs * max_steps) + '_steps.pkl'

    if os.path.isfile(savefile):
        os.remove(savefile)

    for i in range(epochs):
        result = trainer.train()

        print(s.format(
        i + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"]))

    action_history_list = []
    with open(savefile, 'rb') as f:
        while True:
            try:
                action_history_list.append(pickle.load(f).tolist())
            except EOFError:
                break

    action_history_array = np.array(action_history_list).transpose()
    for i in range(num_agents):
        env.action_history[env.players[i]] = action_history_array[i].tolist()
else:
    # Q-learning

    # Hyperparameters
    alpha = 0.05
    beta = 0.2
    delta = 0.99

    q_learner = Q_Learner(env, num_agents=num_agents, m=m, alpha=alpha, beta=beta, delta=delta, epochs=epochs)

    q_learner.train()
    q_learner.plot(last_n=1000)
    q_learner.plot(last_n=100)

    observation = env.deviate(direction='down')
    q_learner.eval(observation, n=10)
    q_learner.plot(last_n=20, title_str='down_deviation_')

    observation = env.deviate(direction='up')
    q_learner.eval(observation, n=10)
    q_learner.plot(last_n=20, title_str='up_deviation_')

env.plot()