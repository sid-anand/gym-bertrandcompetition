from gym.envs.registration import register

register(
    id='BertrandCompetitionMarketDiscrete',
    entry_point='gym_foo.envs:BertrandCompetitionMarketDiscreteEnv',
)
register(
    id='BertrandCompetitionMarketContinuous',
    entry_point='gym_foo.envs:BertrandCompetitionMarketContinuousEnv',
)
