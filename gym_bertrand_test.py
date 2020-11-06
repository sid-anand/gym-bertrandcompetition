import gym
import gym_bertrandcompetition
from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv

import ray
import numpy as np
from ray.tune.registry import register_env
import ray.rllib.agents.a3c as a3c
from ray.rllib.agents.dqn import DQNTrainer
# from ray.rllib.agents.ppo import PPOAgent
from ray.tune.logger import pretty_print

# cd OneDrive/Documents/Research/gym-bertrandcompetition

config = {
    'env_config': {
        'num_agents': 2,
    },
    'env': 'Bertrand',
    'num_workers': 2,
    # 'eager': True,
    # 'use_pytorch': False,
    'train_batch_size': 200,
    'rollout_fragment_length': 200,
    'lr': 0.001
}

register_env('Bertrand', lambda env_config: BertrandCompetitionDiscreteEnv(num_agents=2))
ray.init(num_cpus=4)
trainer = DQNTrainer(config = config, env = 'Bertrand')

# trainer = PPOAgent(config = config, env = 'Bertrand')

s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"

for i in range(20):
    result = trainer.train()

    print(s.format(
    i + 1,
    result["episode_reward_min"],
    result["episode_reward_mean"],
    result["episode_reward_max"],
    result["episode_len_mean"]))