import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='BertrandCompetitionDiscrete-v0',
    entry_point='gym_bertrandcompetition.envs:BertrandCompetitionDiscreteEnv',
)
register(
    id='BertrandCompetitionContinuous-v0',
    entry_point='gym_bertrandcompetition.envs:BertrandCompetitionContinuousEnv',
)
