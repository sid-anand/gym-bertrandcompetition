import gym
import gym_bertrandcompetition
from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv
from gym_bertrandcompetition.envs.bertrand_competition_continuous import BertrandCompetitionContinuousEnv
from agents.q_learner import Q_Learner

import os
import pickle
import ray
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

# cd OneDrive/Documents/Research/gym-bertrandcompetition

# print('gym', gym.__version__)
# print('ray', ray.__version__)
# print('np', np.__version__)
# print('tf', tf.__version__)

# CHANGE PARAMETERS FOR TESTING
# Parameters
num_agents = 2
k = 1
m = 15
max_steps = 1000000000
convergence = 100000
sessions = 1
state_space = 'discrete' # 'discrete' or 'continuous'
use_pickle = False
# choose from QL, DQN, PPO, A3C
trainer_choice = 'DQN'

if state_space == 'discrete':
    env = BertrandCompetitionDiscreteEnv(num_agents=num_agents, k=k, m=m, max_steps=max_steps, sessions=sessions, convergence=convergence, trainer_choice=trainer_choice, use_pickle=use_pickle)
elif state_space == 'continuous':
    env = BertrandCompetitionContinuousEnv(num_agents=num_agents, k=k, max_steps=max_steps, sessions=sessions, trainer_choice=trainer_choice, use_pickle=use_pickle)

config = {
    'env_config': {
        'num_agents': num_agents,
    },
    'env': 'Bertrand',
    'num_workers': num_agents,
    'train_batch_size': 200, # Does this limit training?
    'rollout_fragment_length': 200, # Does this limit training?
    'explore': True, # Change this to False to evaluate (https://docs.ray.io/en/master/rllib-training.html)
    'monitor': True,
    'log_level': 'WARN', # Change this to 'INFO' for more information
    'lr': 0.001
} # Perhaps specify to use GPU in config? (https://docs.ray.io/en/latest/using-ray-with-gpus.html)

if trainer_choice != 'QL':
    register_env('Bertrand', lambda env_config: env)
    ray.init(num_cpus=4)

    if trainer_choice == 'DQN':
        from ray.rllib.agents.dqn import DQNTrainer
        trainer = DQNTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'PPO':
        from ray.rllib.agents.ppo import PPOTrainer
        trainer = PPOTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'A3C':
        from ray.rllib.agents.a3c import A3CTrainer
        trainer = A3CTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'MADDPG':
        from ray.rllib.contrib.maddpg import MADDPGTrainer
        trainer = MADDPGTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'DDPG':
        from ray.rllib.agents.ddpg import DDPGTrainer
        trainer = DDPGTrainer(config = config, env = 'Bertrand')

    s = "Epoch {:3d} / Reward Min: {:6.2f} / Mean: {:6.2f} / Max: {:6.2f} / Steps {:6.2f}"

    savefile = './arrays/' + state_space + '_' + trainer_choice + '_with_' + str(num_agents) + '_agents_k_' + str(k) + '_for_' + str(sessions) + '_sessions.pkl'

    if use_pickle and os.path.isfile(savefile):
        os.remove(savefile)

    for i in range(sessions):
        result = trainer.train()

        print(s.format(
        i + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"]))

    if use_pickle:
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

        env.plot()
        env.plot_last(last_n=1000)
        env.plot_last(last_n=100)

        os.remove(savefile)
else:
    # Q-learning

    # Hyperparameters
    alpha = 0.05
    beta = 0.000002
    delta = 0.95
    log_frequency = 10000

    q_learner = Q_Learner(env, num_agents=num_agents, m=m, alpha=alpha, beta=beta, delta=delta, sessions=sessions, log_frequency=log_frequency)

    q_learner.train()

    env.plot()
    env.plot_last(last_n=1000)
    env.plot_last(last_n=100)

    observation = env.deviate(direction='down')
    q_learner.eval(observation, n=10)
    env.plot_last(last_n=20, title_str='_down_deviation')

    observation = env.deviate(direction='up')
    q_learner.eval(observation, n=10)
    env.plot_last(last_n=20, title_str='_up_deviation')