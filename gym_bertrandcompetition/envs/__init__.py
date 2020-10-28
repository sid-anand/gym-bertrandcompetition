# from gym_bertrandcompetition.envs.bertrand_competition_discrete import BertrandCompetitionDiscreteEnv
# from gym_bertrandcompetition.envs.bertrand_competition_continuous import BertrandCompetitionContinuousEnv
from bertrand_competition_discrete import BertrandCompetitionDiscreteEnv
# from bertrand_competition_continuous import BertrandCompetitionContinuousEnv

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, PPOAgent

register_env("bertrand_competition_discrete", lambda c: BertrandCompetitionDiscreteEnv)
trainer = PPOAgent(env="bertrand_competition_discrete")
while True:
    print(trainer.train())  # distributed training step