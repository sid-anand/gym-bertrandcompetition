import gym
import gym_bertrandcompetition
from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv
from gym_bertrandcompetition.envs.bertrand_competition_continuous import BertrandCompetitionContinuousEnv
from agents.q_learner import Q_Learner
from agents.sarsa import SARSA

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

# Trainer Choice (Options: QL, SARSA, DQN, PPO, A3C, A2C, DDPG)
trainer_choice = 'DQN'
mitigation_agent = True

# Parameters
num_agents = 2
k = 0
m = 15
convergence = 100000
sessions = 1

# Hyperparameters
alpha = 0.15 # Change these to test Calvano results
beta = 0.00001 # Change these to test Calvano results
delta = 0.95
log_frequency = 50000

# Performance and Testing
num_gpus = 0
overwrite_id = 2
len_eval_after_deviation = 20

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
    # Change 'log_level' to 'INFO' for more information
    'gamma': delta
}

path = os.path.abspath(os.getcwd())

def eval_then_unload(observation):
    for i in range(len_eval_after_deviation):
        # action = trainer.compute_action(observation)
        action = {}
        for agent_id, agent_obs in observation.items():
            policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
            action[agent_id] = trainer.compute_action(agent_obs, policy_id=policy_id)
            # action[agent_id] = trainer.compute_action(agent_obs) # From before multi-agent integration
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
        env.action_history[env.agents[i]].extend(action_history_array[i].tolist())




if trainer_choice not in ['QL', 'SARSA']:

    use_pickle = True
    max_steps = 200000

    if trainer_choice in ['DQN', 'PPO']:
        state_space = 'discrete'
        env = BertrandCompetitionDiscreteEnv(num_agents=num_agents, k=k, m=m, max_steps=max_steps, sessions=sessions, convergence=convergence, trainer_choice=trainer_choice, mitigation_agent=mitigation_agent, use_pickle=use_pickle, path=path)
    else:
        state_space = 'continuous'
        env = BertrandCompetitionContinuousEnv(num_agents=num_agents, k=k, max_steps=max_steps, sessions=sessions, trainer_choice=trainer_choice, mitigation_agent=mitigation_agent, use_pickle=use_pickle, path=path)

    multiagent_dict = dict()
    multiagent_policies = dict()

    for agent in env.agents:
        agent_entry = (
            None,
            env.observation_spaces[agent],
            env.action_spaces[agent],
            {}
        )
        multiagent_policies[agent] = agent_entry

    multiagent_dict['policies'] = multiagent_policies
    multiagent_dict['policy_mapping_fn'] = lambda agent_id: agent_id
    config['multiagent'] = multiagent_dict

    savefile = './arrays/' + state_space + '_' + trainer_choice + '_with_' + str(num_agents) + '_agents_k_' + str(k) + '_for_' + str(sessions) + '_sessions.pkl'

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
                "epsilon_timesteps": 250000,  # Timesteps over which to anneal epsilon.
            }
        trainer = DQNTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'PPO':
        from ray.rllib.agents.ppo import PPOTrainer
        config['num_workers'] = num_agents
        trainer = PPOTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'A3C':
        from ray.rllib.agents.a3c import A3CTrainer
        trainer = A3CTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'A2C':
        from ray.rllib.agents.a3c import A2CTrainer
        trainer = A2CTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'MADDPG':
        from ray.rllib.contrib.maddpg import MADDPGTrainer
        config['agent_id'] = 0
        trainer = MADDPGTrainer(config = config, env = 'Bertrand')
    elif trainer_choice == 'DDPG':
        from ray.rllib.agents.ddpg import DDPGTrainer
        trainer = DDPGTrainer(config = config, env = 'Bertrand')

    if use_pickle and os.path.isfile(savefile):
        os.remove(savefile)

    analysis = tune.run(
        trainer_choice, 
        # num_samples = 4,
        config = config, 
        local_dir = './log', 
        stop = {'training_iteration': sessions},
        mode = 'max',
        metric = 'episode_reward_mean',
        checkpoint_at_end = True
    )

    trainer.restore(analysis.best_checkpoint)

    # s = "Epoch {:3d} / Reward Min: {:6.2f} / Mean: {:6.2f} / Max: {:6.2f} / Steps {:6.2f}"

    # for i in range(sessions):
    #     result = trainer.train()

    #     print(s.format(
    #     i + 1,
    #     result["episode_reward_min"],
    #     result["episode_reward_mean"],
    #     result["episode_reward_max"],
    #     result["episode_len_mean"]))

    #     checkpoint = trainer.save()
    #     print('Checkpoint: ', checkpoint)

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
            env.action_history[env.agents[i]] = action_history_array[i].tolist()

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

    max_steps = 2000000
    state_space = 'discrete'
    use_pickle = False

    env = BertrandCompetitionDiscreteEnv(num_agents=num_agents, k=k, m=m, max_steps=max_steps, sessions=sessions, convergence=convergence, trainer_choice=trainer_choice, mitigation_agent=mitigation_agent, use_pickle=use_pickle, path=path)

    if trainer_choice == 'QL':
        trainer = Q_Learner(env, num_agents=num_agents, m=m, alpha=alpha, beta=beta, delta=delta, sessions=sessions, log_frequency=log_frequency)
    elif trainer_choice == 'SARSA':
        trainer = SARSA(env, num_agents=num_agents, m=m, alpha=alpha, beta=beta, delta=delta, sessions=sessions, log_frequency=log_frequency)

    trainer.train()

    with open('./q_tables/' + state_space + '_' + trainer_choice + '_with_' + str(num_agents) + '_agents_k_' + str(k) + '_for_' + str(sessions) + '_sessions.pkl', 'wb') as f:
        pickle.dump(trainer.q_table, f)

    env.plot(overwrite_id=overwrite_id)
    env.plot_last(last_n=1000, overwrite_id=overwrite_id)
    env.plot_last(last_n=100, overwrite_id=overwrite_id)

    observation = env.deviate(direction='down')
    trainer.eval(observation, n=len_eval_after_deviation)
    env.plot_last(last_n=25, title_str='_down_deviation', overwrite_id=overwrite_id)

    observation = env.deviate(direction='up')
    trainer.eval(observation, n=len_eval_after_deviation)
    env.plot_last(last_n=25, title_str='_up_deviation', overwrite_id=overwrite_id)