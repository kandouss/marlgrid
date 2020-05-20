from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class VisibilityTestEnv(MultiGridEnv):
    mission = ""
    metadata = {}

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        self.grid.horz_wall(0, height // 2, width - 3, obj_type=Wall)

        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)
