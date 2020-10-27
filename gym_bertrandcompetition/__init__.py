import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='BertrandCompetitionDiscrete',
    entry_point='gym_bertrandcompetition.envs:BertrandCompetitionDiscreteEnv',
)
register(
    id='BertrandCompetitionContinuous',
    entry_point='gym_bertrandcompetition.envs:BertrandCompetitionContinuousEnv',
)
