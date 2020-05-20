from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class ClutteredMultiGrid(MultiGridEnv):
    mission = "get to the green square"
    metadata = {}

    def __init__(self, *args, n_clutter=25, randomize_goal=False, **kwargs):
        self.n_clutter = n_clutter
        self.randomize_goal = randomize_goal
        super().__init__(*args, **kwargs)

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        if self.randomize_goal:
            self.place_obj(Goal(color="green", reward=1), max_tries=100)
        else:
            self.put_obj(Goal(color="green", reward=1), width - 2, height - 2)
        for _ in range(self.n_clutter):
            self.place_obj(Wall(), max_tries=100)

        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)