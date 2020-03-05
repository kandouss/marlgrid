__all__ = ['env', 'utils']

from gym.envs.registration import register as register_gym_environment
from .env.envs import *
from .env.agents import InteractiveAgent

def make_marl_env(env_class, n_agents, *args, **kwargs):
    return env_class(
        agents=[InteractiveAgent(color='red', view_size=5)
            for _ in range(n_agents)],
        *args,
        **kwargs
    )

register_gym_environment(
    id='MarlGrid-3AgentEmpty9x9-v0',
    entry_point = make_marl_env(EmptyMultiGrid, n_agents=3, grid_size=9, done_condition='all')
)