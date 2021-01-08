import gym
import gym_bertrandcompetition
from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv
from gym_bertrandcompetition.envs.bertrand_competition_continuous import BertrandCompetitionContinuousEnv
from agents.q_learner import Q_Learner

import os
import pickle
import ray
from ray import tune
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

# CHANGE PARAMETERS FOR TESTING
# Parameters
num_agents = 2
k = 1
m = 15
max_steps = 100000 # 1000000000 from Calvano paper
convergence = 100000
sessions = 1
state_space = 'discrete' # 'discrete' or 'continuous'

use_pickle = True
num_gpus = 0
overwrite_id = 2
len_eval_after_deviation = 20
# choose from QL, DQN, PPO, A3C, DDPG, MADDPG
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
    'num_gpus': num_gpus,
    'train_batch_size': 200, # Does this limit training?
    'rollout_fragment_length': 200, # Does this limit training?
    'batch_mode': 'complete_episodes',
    # Change 'explore' to True to False to evaluate (https://docs.ray.io/en/master/rllib-training.html)
    # 'monitor': True,
    # Change 'log_level' to 'INFO' for more information
    'gamma': 0.95
} # Perhaps specify to use GPU in config? (https://docs.ray.io/en/latest/using-ray-with-gpus.html)

savefile = './arrays/' + state_space + '_' + trainer_choice + '_with_' + str(num_agents) + '_agents_k_' + str(k) + '_for_' + str(sessions) + '_sessions.pkl'

def eval_then_unload(observation):
    for i in range(len_eval_after_deviation):
        # action = trainer.compute_action(observation)
        action = {}
        for agent_id, agent_obs in observation.items():
            # policy_id = self.config['multiagent']['policy_mapping_fn'](agent_id)
            # action[agent_id] = self.agent.compute_action(agent_obs, policy_id=policy_id) # Does this imply I'm using the same policy for both agents?
            action[agent_id] = trainer.compute_action(agent_obs)
        observation, _, _, _ = env.step(action)

    action_history_list = []
    with open(savefile, 'rb') as f:
        while True:
            try:
                action_history_list.append(pickle.load(f).tolist())
            except EOFError:
                break

    action_history_array = np.array(action_history_list).transpose()
    for i in range(num_agents):
        env.action_history[env.players[i]].extend(action_history_array[i].tolist())




if trainer_choice != 'QL':
    register_env('Bertrand', lambda env_config: env)
    ray.init(num_cpus=4)

    if trainer_choice == 'DQN':
        from ray.rllib.agents.dqn import DQNTrainer
        config['exploration_config'] = {
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 1.0,
                "final_epsilon": 0.000001,
                "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
            }
        trainer = DQNTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'PPO':
        from ray.rllib.agents.ppo import PPOTrainer
        config['num_workers'] = num_agents
        trainer = PPOTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'A3C':
        from ray.rllib.agents.a3c import A3CTrainer
        trainer = A3CTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'MADDPG':
        from ray.rllib.contrib.maddpg import MADDPGTrainer
        config['agent_id'] = 0
        trainer = MADDPGTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'DDPG':
        from ray.rllib.agents.ddpg import DDPGTrainer
        trainer = DDPGTrainer(config = config, env = 'Bertrand')

    s = "Epoch {:3d} / Reward Min: {:6.2f} / Mean: {:6.2f} / Max: {:6.2f} / Steps {:6.2f}"

    if use_pickle and os.path.isfile(savefile):
        os.remove(savefile)

    # analysis = tune.run(trainer_choice, config=config, local_dir='./log', checkpoint_at_end=True)
    # checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial("episode_reward_mean"), metric="episode_reward_mean")

    for i in range(sessions):
        result = trainer.train()

        print(s.format(
        i + 1,
        result["episode_reward_min"],
        result["episode_reward_mean"],
        result["episode_reward_max"],
        result["episode_len_mean"]))

        # checkpoint = trainer.save()
        # print('Checkpoint: ', checkpoint)

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

        env.plot(overwrite_id=overwrite_id)
        env.plot_last(last_n=1000, overwrite_id=overwrite_id)
        env.plot_last(last_n=100, overwrite_id=overwrite_id)

        # Deviate downwards
        observation = env.deviate(direction='down')
        eval_then_unload(observation)
        env.plot_last(last_n=30, title_str='_down_deviation', overwrite_id=overwrite_id)

        # Deviate upwards
        observation = env.deviate(direction='up')
        eval_then_unload(observation)
        env.plot_last(last_n=30, title_str='_up_deviation', overwrite_id=overwrite_id)

        os.remove(savefile)
else:
    # Q-learning

    # Hyperparameters
    alpha = 0.15 # Change these to test Calvano results
    beta = 0.00001 # Change these to test Calvano results
    delta = 0.95
    log_frequency = 50000

    q_learner = Q_Learner(env, num_agents=num_agents, m=m, alpha=alpha, beta=beta, delta=delta, sessions=sessions, log_frequency=log_frequency)

    q_learner.train()

    with open('./q_tables/' + savefile + '.pkl', 'wb') as f:
        pickle.dump(self.q_table, f)

    env.plot(overwrite_id=overwrite_id)
    env.plot_last(last_n=1000, overwrite_id=overwrite_id)
    env.plot_last(last_n=100, overwrite_id=overwrite_id)

    observation = env.deviate(direction='down')
    q_learner.eval(observation, n=len_eval_after_deviation)
    env.plot_last(last_n=25, title_str='_down_deviation', overwrite_id=overwrite_id)

    observation = env.deviate(direction='up')
    q_learner.eval(observation, n=len_eval_after_deviation)
    env.plot_last(last_n=25, title_str='_up_deviation', overwrite_id=overwrite_id)