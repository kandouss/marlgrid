from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class ClutteredGoalCycleEnv(MultiGridEnv):
    mission = "Cycle between yellow goal tiles."
    metadata = {}

    def __init__(self, *args, reward=1, penalty=0.0, n_clutter=None, clutter_density=None, n_bonus_tiles=3, **kwargs):
        if (n_clutter is None) == (clutter_density is None):
            raise ValueError("Must provide n_clutter xor clutter_density in environment config.")

        super().__init__(*args, **kwargs)

        if clutter_density is not None:
            self.n_clutter = int(clutter_density * (self.width-2)*(self.height-2))
        else:
            self.n_clutter = n_clutter
        self.reward = reward
        self.penalty = penalty

        self.n_bonus_tiles = n_bonus_tiles

        self.reset()


    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        for bonus_id in range(self.n_bonus_tiles):
            self.place_obj(BonusTile(color="yellow", reward=self.reward, penalty=self.penalty, bonus_id=bonus_id, n_bonus=self.n_bonus_tiles), max_tries=100)
        for _ in range(self.n_clutter):
            self.place_obj(Wall(), max_tries=100)

        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)
