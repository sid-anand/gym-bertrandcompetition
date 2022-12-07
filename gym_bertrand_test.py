# import gym
# import gym_bertrandcompetition
from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv
# from gym_bertrandcompetition.envs.bertrand_competition_continuous import BertrandCompetitionContinuousEnv
from agents.q_learner import Q_Learner
# from agents.sarsa import SARSA
# from agents.combo_multiagent import custom_training_workflow_ppo_dqn
# from agents.combo_multiagent import custom_training_workflow_ppo_a3c
# from agents.combo_multiagent import custom_training_workflow_dqn_a3c
# from agents.combo_multiagent import custom_training_workflow_ppo_ddpg

##################################################

# import argparse
# import gym
import os

# import ray
# from ray import tune
# import pandas as pd
# from ray.rllib.agents.trainer_template import build_trainer
# from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG as DQN_CONFIG
# from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
# from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy
# from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
# from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
# from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
# from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as A3C_CONFIG
# from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
# from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
# from ray.rllib.agents.ddpg.ddpg import DEFAULT_CONFIG as DDPG_CONFIG
# from ray.rllib.agents.ddpg.ddpg_tf_policy import DDPGTFPolicy
# from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
# from ray.rllib.evaluation.worker_set import WorkerSet
# from ray.rllib.execution.common import _get_shared_metrics
# from ray.rllib.execution.concurrency_ops import Concurrently
# from ray.rllib.execution.metric_ops import StandardMetricsReporting
# from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, \
#     StandardizeFields, SelectExperiences
# from ray.rllib.execution.replay_ops import StoreToReplayBuffer, Replay
# from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
# from ray.rllib.execution.replay_buffer import LocalReplayBuffer
# from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
# from ray.rllib.utils.test_utils import check_learning_achieved
# from ray.tune.registry import register_env

##################################################

# import os
# import pickle
# import ray
# from ray import tune
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from ray.tune.registry import register_env
# from ray.tune.logger import pretty_print

# CHANGE PARAMETERS FOR TESTING

# Trainer Choice (Options: QL, SARSA, DQN, PPO, A3C, A2C, DDPG)
trainer_choice = 'QL'
second_trainer_choice = ''  # leave as empty string ('') for none

# Collusion Mitigation Mechanism
supervisor = False
proportion_boost = 1.25

# Parameters
num_agents = 2
c = [0.5, 1]
a_minus_c = [1, 1]
k = 1
m = 15
convergence = 100000
sessions = 1

# Hyperparameters
alpha = 0.1  # Change these to test Calvano results
beta = 0.000005  # Change these to test Calvano results
delta = 0.95
log_frequency = 50000
dqn_epsilon_timesteps = 150000
on_policy_steps = 20000

# Performance and Testing
overwrite_id = 0
num_gpus = 0
len_eval_after_training = 1000
len_eval_after_deviation = 20

if trainer_choice in ['QL', 'SARSA', 'DQN', 'A2C', 'A3C', 'PPO']:
    state_space = 'discrete'
else:
    state_space = 'continuous'

# Savefile
savefile = state_space + '_' + trainer_choice
if second_trainer_choice:
    savefile += '_' + second_trainer_choice
savefile += '_' + str(num_agents) + '_agents_k_' + str(k)
if supervisor:
    savefile += '_supervisor_' + \
        str(supervisor) + '_' + str(proportion_boost).replace('.', '_')
if trainer_choice in ['QL', 'SARSA']:
    savefile += '_alpha_' + str(alpha).replace('.',
                                               '_') + '_beta_' + str(beta).replace('.', '_')
elif trainer_choice == 'DQN' or second_trainer_choice == 'DQN':
    savefile += '_epstep_' + str(dqn_epsilon_timesteps)
elif trainer_choice == 'A3C' or trainer_choice == 'PPO':
    savefile += '_onpolsteps_' + str(on_policy_steps)

config = {
    'env_config': {
        'num_agents': num_agents,
    },
    'env': 'Bertrand',
    'num_gpus': num_gpus,
    'train_batch_size': 200,
    'rollout_fragment_length': 200,
    'batch_mode': 'complete_episodes',
    # Change 'explore' to True to False to evaluate (https://docs.ray.io/en/master/rllib-training.html)
    # 'monitor': True,
    'log_level': 'WARN',  # Change 'log_level' to 'INFO' for more information
    'gamma': delta
}

path = os.path.abspath(os.getcwd())


# def eval_then_unload(observation, len_eval):
#     '''Used to compute actions for certain observations and unload the results that are automatically pickled.'''
#     for i in range(len_eval):
#         action = {}
#         for agent_id, agent_obs in observation.items():
#             policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
#             action[agent_id] = trainer.compute_action(
#                 observation=agent_obs, policy_id=policy_id)
#         observation, _, _, _ = env.step(action)

#     action_history_list = []
#     with open(pklfile, 'rb') as f:
#         while True:
#             try:
#                 action_history_list.append(pickle.load(f).tolist())
#             except EOFError:
#                 break

#     action_history_array = np.array(action_history_list).transpose()
#     for i in range(num_agents):
#         env.action_history[env.agents[i]].extend(
#             action_history_array[i].tolist())


# if trainer_choice not in ['QL', 'SARSA']:
#     # RLLib Algorithms

#     use_pickle = True
#     if trainer_choice == 'DQN' or second_trainer_choice == 'DQN':
#         max_steps = dqn_epsilon_timesteps + 5000
#     elif trainer_choice == 'A3C' or trainer_choice == 'PPO':
#         max_steps = on_policy_steps
#         len_eval_after_training = 2500
#     else:
#         max_steps = 50000

#     pklfile = './arrays/' + savefile + '.pkl'

#     if os.path.isfile(pklfile):
#         os.remove(pklfile)

#     if state_space == 'discrete':
#         env = BertrandCompetitionDiscreteEnv(
#             num_agents=num_agents,
#             k=k,
#             m=m,
#             max_steps=max_steps,
#             sessions=sessions,
#             convergence=convergence,
#             trainer_choice=trainer_choice,
#             supervisor=supervisor,
#             proportion_boost=proportion_boost,
#             use_pickle=use_pickle,
#             path=path,
#             savefile=savefile
#         )
#     else:
#         env = BertrandCompetitionContinuousEnv(
#             num_agents=num_agents,
#             k=k,
#             max_steps=max_steps,
#             sessions=sessions,
#             trainer_choice=trainer_choice,
#             supervisor=supervisor,
#             proportion_boost=proportion_boost,
#             use_pickle=use_pickle,
#             path=path,
#             savefile=savefile
#         )

#     if not second_trainer_choice:
#         # Single algorithm training

#         multiagent_dict = dict()
#         multiagent_policies = dict()

#         for agent in env.agents:
#             agent_entry = (
#                 None,
#                 env.observation_spaces[agent],
#                 env.action_spaces[agent],
#                 {}
#             )
#             multiagent_policies[agent] = agent_entry

#         if supervisor:
#             agent_entry = (
#                 None,
#                 env.observation_spaces['supervisor'],
#                 env.action_spaces['supervisor'],
#                 {}
#             )
#             multiagent_policies['supervisor'] = agent_entry

#         multiagent_dict['policies'] = multiagent_policies
#         multiagent_dict['policy_mapping_fn'] = lambda agent_id: agent_id
#         config['multiagent'] = multiagent_dict

#         register_env('Bertrand', lambda env_config: env)
#         ray.init(num_cpus=4)

#         if trainer_choice == 'DQN':
#             from ray.rllib.agents.dqn import DQNTrainer
#             config['exploration_config'] = {
#                 # The Exploration class to use.
#                 "type": "EpsilonGreedy",
#                 # Config for the Exploration class' constructor:
#                 "initial_epsilon": 1.0,
#                 "final_epsilon": 0.000001,
#                 # Timesteps over which to anneal epsilon. Originally set to 250000.
#                 "epsilon_timesteps": dqn_epsilon_timesteps,
#             }
#             # For eval afterward
#             config_copy = config.copy()
#             config_copy['explore'] = False
#             trainer = DQNTrainer(config=config_copy, env='Bertrand')
#         elif trainer_choice == 'PPO':
#             from ray.rllib.agents.ppo import PPOTrainer
#             config['num_workers'] = 1
#             # config['lr'] = 0.001
#             # For eval afterward
#             config_copy = config.copy()
#             config_copy['explore'] = False
#             trainer = PPOTrainer(config=config_copy, env='Bertrand')
#         elif trainer_choice == 'A3C':
#             from ray.rllib.agents.a3c import A3CTrainer
#             config['num_workers'] = 1
#             # config['lr'] = 0.01
#             # For eval afterward
#             config_copy = config.copy()
#             config_copy['explore'] = False
#             trainer = A3CTrainer(config=config_copy, env='Bertrand')
#         elif trainer_choice == 'A2C':
#             from ray.rllib.agents.a3c import A2CTrainer
#             config['num_workers'] = 1
#             # For eval afterward
#             config_copy = config.copy()
#             config_copy['explore'] = False
#             trainer = A2CTrainer(config=config_copy, env='Bertrand')
#         elif trainer_choice == 'MADDPG':
#             from ray.rllib.contrib.maddpg import MADDPGTrainer
#             config['agent_id'] = 0
#             # For eval afterward
#             config_copy = config.copy()
#             config_copy['explore'] = False
#             trainer = MADDPGTrainer(config=config_copy, env='Bertrand')
#         elif trainer_choice == 'DDPG':
#             from ray.rllib.agents.ddpg import DDPGTrainer
#             # For eval afterward
#             config_copy = config.copy()
#             config_copy['explore'] = False
#             trainer = DDPGTrainer(config=config_copy, env='Bertrand')

#         analysis = tune.run(
#             trainer_choice,
#             # num_samples = 4,
#             config=config,
#             local_dir='./log',
#             stop={'training_iteration': sessions},
#             mode='max',
#             metric='episode_reward_mean',
#             checkpoint_at_end=True
#         )

#         trainer.restore(checkpoint_path=analysis.best_checkpoint)

#         # analysis = tune.run(
#         #     trainer_choice,
#         #     # num_samples = 4,
#         #     config = config_copy,
#         #     local_dir = './log',
#         #     stop = {'training_iteration': sessions},
#         #     mode = 'max',
#         #     metric = 'episode_reward_mean',
#         #     restore = analysis.best_checkpoint,
#         #     checkpoint_at_end = True
#         # )
#     else:
#         # Dual algorithm training

#         register_env('Bertrand', lambda env_config: env)
#         ray.init(num_cpus=4)

#         if state_space == 'discrete':
#             policies = {
#                 # change this later
#                 "PPO_policy": (PPOTFPolicy, env.observation_spaces['agent_0'], env.action_spaces['agent_0'], PPO_CONFIG),
#                 # change this later
#                 "DQN_policy": (DQNTFPolicy, env.observation_spaces['agent_0'], env.action_spaces['agent_0'], DQN_CONFIG),
#                 "A3C_policy": (A3CTFPolicy, env.observation_spaces['agent_0'], env.action_spaces['agent_0'], A3C_CONFIG)
#             }
#         else:
#             policies = {
#                 "PPO_policy": (PPOTFPolicy, env.observation_spaces['agent_0'], env.action_spaces['agent_0'], PPO_CONFIG),
#                 "A3C_policy": (A3CTFPolicy, env.observation_spaces['agent_0'], env.action_spaces['agent_0'], A3C_CONFIG),
#                 "DDPG_policy": (DDPGTFPolicy, env.observation_spaces['agent_0'], env.action_spaces['agent_0'], DDPG_CONFIG)
#             }

#         policies_to_train_list = [trainer_choice +
#                                   '_policy', second_trainer_choice + '_policy']

#         def policy_mapping_fn(agent_id):
#             if agent_id == 'agent_0':
#                 return trainer_choice + '_policy'
#             else:
#                 return second_trainer_choice + '_policy'

#         trainer_choice_list = [trainer_choice, second_trainer_choice]
#         if 'PPO' in trainer_choice_list and 'DQN' in trainer_choice_list:
#             custom_training_workflow = custom_training_workflow_ppo_dqn
#         elif 'PPO' in trainer_choice_list and 'A3C' in trainer_choice_list:
#             custom_training_workflow = custom_training_workflow_ppo_a3c
#         elif 'DQN' in trainer_choice_list and 'A3C' in trainer_choice_list:
#             custom_training_workflow = custom_training_workflow_dqn_a3c
#         elif 'PPO' in trainer_choice_list and 'DDPG' in trainer_choice_list:
#             custom_training_workflow = custom_training_workflow_ppo_ddpg

#         trainer = build_trainer(
#             name=trainer_choice + '_' + second_trainer_choice + '_MultiAgent',
#             default_policy=None,
#             execution_plan=custom_training_workflow)

#         config['multiagent'] = {
#             "policies": policies,
#             "policy_mapping_fn": policy_mapping_fn,
#             "policies_to_train": policies_to_train_list,
#         }

#         analysis = tune.run(
#             trainer,
#             # num_samples = 4,
#             config=config,
#             local_dir='./log',
#             stop={'training_iteration': sessions},
#             mode='max',
#             metric='episode_reward_mean',
#             checkpoint_at_end=True
#         )

#         # trainer.restore(checkpoint_path=analysis.best_checkpoint)

#     action_history_list = []
#     with open(pklfile, 'rb') as f:
#         while True:
#             try:
#                 action_history_list.append(pickle.load(f).tolist())
#             except EOFError:
#                 break

#     action_history_array = np.array(action_history_list).transpose()
#     for i in range(num_agents):
#         env.action_history[env.agents[i]] = action_history_array[i].tolist()

#     env.plot(overwrite_id=overwrite_id)
#     env.plot_last(last_n=100, title_str='_train', overwrite_id=overwrite_id)
#     env.plot_last(last_n=1000, window=100, title_str='_train',
#                   overwrite_id=overwrite_id)

#     # if supervisor:
#     #     demand_change = []
#     #     with open('./arrays/' + savefile + 'demand.pkl', 'rb') as f:
#     #         while True:
#     #             try:
#     #                 demand_change.append(pickle.load(f).tolist())
#     #             except EOFError:
#     #                 break

#     #     demand_change_array = np.array(demand_change).transpose()

#     #     x = np.arange(len(demand_change_array[0]))
#     #     plt.plot(x, demand_change_array[0], 'c', label='agent_0', alpha=0.75)
#     #     plt.plot(x, demand_change_array[1], 'y', label='agent_1', alpha=0.75)
#     #     plt.plot(x, pd.Series(demand_change_array[0]).rolling(window=1000).mean(), alpha=0.5, label='agent_0 MA')
#     #     plt.plot(x, pd.Series(demand_change_array[1]).rolling(window=1000).mean(), alpha=0.5, label='agent_1 MA')
#     #     plt.xlabel('Steps')
#     #     plt.ylabel('Absolute Demand Change')
#     #     plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#     #     plt.legend()
#     #     plt.savefig('figures/demand_error' + str(overwrite_id))
#     #     plt.clf()

#     if not second_trainer_choice:
#         # Eval
#         observation = env.one_step()
#         eval_then_unload(observation=observation,
#                          len_eval=len_eval_after_training)
#         env.plot_last(last_n=len_eval_after_training,
#                       window=100, overwrite_id=overwrite_id)

#         # Deviate downwards
#         observation = env.deviate(direction='down')
#         eval_then_unload(observation=observation,
#                          len_eval=len_eval_after_deviation)
#         env.plot_last(last_n=30, title_str='_down_deviation',
#                       overwrite_id=overwrite_id)

#         # Deviate upwards
#         observation = env.deviate(direction='up')
#         eval_then_unload(observation=observation,
#                          len_eval=len_eval_after_deviation)
#         env.plot_last(last_n=30, title_str='_up_deviation',
#                       overwrite_id=overwrite_id)

#     os.remove(pklfile)

# else:
# Algorithms from scratch

max_steps = 2500000
# for alpha = 0.15 beta = 0.00001 its 1500000,
# for alpha = 0.1 beta = 0.000005 its 2500000,
# for alpha = 0.075 beta = 0.0000025 its 4000000,
# for alpha = 0.05 beta = 0.0000025 its 4000000
use_pickle = False

env = BertrandCompetitionDiscreteEnv(
    num_agents=num_agents,
    c=c,
    a_minus_c=a_minus_c,
    k=k,
    m=m,
    max_steps=max_steps,
    sessions=sessions,
    convergence=convergence,
    trainer_choice=trainer_choice,
    supervisor=False,
    proportion_boost=proportion_boost,
    use_pickle=use_pickle,
    path=path,
    savefile=savefile
)

# if trainer_choice == 'QL':
trainer = Q_Learner(
    env=env,
    num_agents=num_agents,
    m=m,
    alpha=alpha,
    beta=beta,
    delta=delta,
    supervisor=supervisor,
    proportion_boost=proportion_boost,
    action_price_space=env.action_price_space,
    sessions=sessions,
    log_frequency=log_frequency
)
# elif trainer_choice == 'SARSA':
#     trainer = SARSA(env=env,
#                     num_agents=num_agents,
#                     m=m,
#                     alpha=alpha,
#                     beta=beta,
#                     delta=delta,
#                     supervisor=supervisor,
#                     sessions=sessions,
#                     log_frequency=log_frequency
#                     )

trainer.train()

env.plot(overwrite_id=overwrite_id)
env.plot_last(last_n=100, title_str='_train', overwrite_id=overwrite_id)

# Price Error Plot

# x = np.arange(len(env.price_error[0]))
# plt.plot(x, env.price_error[0], 'c', label='agent_0', alpha=0.75)
# plt.plot(x, env.price_error[1], 'y', label='agent_1', alpha=0.75)
# plt.plot(x, pd.Series(env.price_error[0]).rolling(window=1000).mean(), alpha=0.5, label='agent_0 MA')
# plt.plot(x, pd.Series(env.price_error[1]).rolling(window=1000).mean(), alpha=0.5, label='agent_1 MA')
# plt.xlabel('Steps')
# plt.ylabel('Absolute Price Change')
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# plt.legend()
# plt.savefig('figures/price_error' + str(overwrite_id))
# plt.clf()

# observation = env.one_step()
# trainer.eval(observation, n=len_eval_after_training)
# env.plot_last(last_n=len_eval_after_training, window=100, overwrite_id=overwrite_id)

observation = env.deviate(direction='down')
trainer.eval(observation, n=len_eval_after_deviation)
env.plot_last(last_n=25, title_str='_down_deviation',
              overwrite_id=overwrite_id)

observation = env.deviate(direction='up')
trainer.eval(observation, n=len_eval_after_deviation)
env.plot_last(last_n=25, title_str='_up_deviation',
              overwrite_id=overwrite_id)
