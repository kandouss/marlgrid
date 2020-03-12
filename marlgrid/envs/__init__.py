from .empty import EmptyMultiGrid
from .doorkey import DoorKeyEnv
from .cluttered import ClutteredMultiGrid
from .viz_test import VisibilityTestEnv

from ..agents import InteractiveAgent
from gym.envs.registration import register as gym_register

import sys

this_module = sys.modules[__name__]
registered_envs = []


def register_marl_env(
    env_name,
    env_class,
    n_agents,
    grid_size,
    view_size,
    view_tile_size=8,
    done_condition="all",
    env_kwargs={},
):
    colors = ["red", "blue", "purple", "orange", "olive"]
    assert n_agents <= len(colors)

    class RegEnv(env_class):
        def __init__(self):
            super().__init__(
                agents=[
                    InteractiveAgent(color=c, view_size=view_size, view_tile_size=8)
                    for c in colors[:n_agents]
                ],
                grid_size=grid_size,
                done_condition=done_condition,
                **env_kwargs,
            )

    env_class_name = f"env_{len(registered_envs)}"
    setattr(this_module, env_class_name, RegEnv)
    registered_envs.append(env_name)
    gym_register(env_name, entry_point=f"marlgrid.envs:{env_class_name}")

register_marl_env(
    "MarlGrid-1AgentCluttered15x15-v0",
    ClutteredMultiGrid,
    n_agents=1,
    grid_size=11,
    view_size=5,
    env_kwargs={'n_clutter':30}
)

register_marl_env(
    "MarlGrid-3AgentCluttered11x11-v0",
    ClutteredMultiGrid,
    n_agents=3,
    grid_size=11,
    view_size=7,
)

register_marl_env(
    "MarlGrid-3AgentCluttered15x15-v0",
    ClutteredMultiGrid,
    n_agents=3,
    grid_size=15,
    view_size=7,
)

register_marl_env(
    "MarlGrid-2AgentEmpty9x9-v0", EmptyMultiGrid, n_agents=2, grid_size=9, view_size=7
)

register_marl_env(
    "MarlGrid-3AgentEmpty9x9-v0", EmptyMultiGrid, n_agents=3, grid_size=9, view_size=7
)

register_marl_env(
    "MarlGrid-4AgentEmpty9x9-v0", EmptyMultiGrid, n_agents=4, grid_size=9, view_size=7
)
