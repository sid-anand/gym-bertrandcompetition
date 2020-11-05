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

# from ray.rllib.env import MultiAgentEnv

# cd OneDrive/Documents/Research/gym-bertrandcompetition

# env = gym.make('BertrandCompetitionDiscrete-v0')

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

# class Rwrapper(gym.RewardWrapper):
#     def __init__(self, env):
#         self.reward_range = env.reward_range
#         super(gym.RewardWrapper, self).__init__(env)
#     def reward(self, reward):
#         return reward
# register_env('Bertrand', lambda env_config: Rwrapper(BertrandCompetitionDiscreteEnv()))


register_env('Bertrand', lambda env_config: BertrandCompetitionDiscreteEnv())
ray.init(num_cpus=4)
trainer = DQNTrainer(config = config, env = 'Bertrand')

# register_env('Bertrand', lambda env_config: BertrandCompetitionDiscreteEnv())
# trainer = PPOAgent(config = config, env = 'Bertrand')

s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f}"

pN = 1
pM = 10
xi = 0.1
m = 15

a_space = np.linspace(pN - xi * (pM - pN), pM + xi * (pM - pN), m)

for i in range(20):
    result = trainer.train()
    # print(pretty_print(result))
    print(s.format(
    i + 1,
    result["episode_reward_min"],
    result["episode_reward_mean"],
    result["episode_reward_max"],
    result["episode_len_mean"]))

    # print(trainer.compute_action({'agent_0': (a_space[3], a_space[3]), 'agent_1': (a_space[3], a_space[3])}))
    # print(trainer.compute_action({'agent_0': (0, 0), 'agent_1': (0, 0)}))
    # print(trainer.compute_action(a_space[3], a_space[3]))