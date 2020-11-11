import gym
import gym_bertrandcompetition
from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv

import ray
import numpy as np
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
max_steps = 500
epochs = 50
plot = True
# choose from DQN, PPO, A3C
trainer_choice = 'DQN'

env = BertrandCompetitionDiscreteEnv(num_agents=num_agents, k=k, max_steps=max_steps, plot=plot, epochs=epochs, trainer_choice=trainer_choice)

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