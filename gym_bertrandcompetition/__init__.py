from gym.envs.registration import register

register(
    id='BertrandCompetitionDiscrete',
    entry_point='gym_foo.envs:BertrandCompetitionDiscreteEnv',
)
register(
    id='BertrandCompetitionContinuous',
    entry_point='gym_foo.envs:BertrandCompetitionContinuousEnv',
)
