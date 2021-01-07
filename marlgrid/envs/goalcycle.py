from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class ClutteredGoalCycleEnv(MultiGridEnv):
    mission = "Cycle between yellow goal tiles."
    metadata = {}

    def __init__(
            self,
            *args,
            reward=1,
            reward_streak_bonus=0,
            penalty=0.0,
            n_clutter=None,
            clutter_density=None,
            n_bonus_tiles=3,
            initial_reward=True,
            cycle_reset=False,
            reset_on_mistake=False,
            reward_decay=False,
            **kwargs,
            ):
        if (n_clutter is None) == (clutter_density is None):
            raise ValueError("Must provide n_clutter xor clutter_density in environment config.")

        # Overwrite the default reward_decay for goal cycle environments.
        super().__init__(*args, **{**kwargs, 'reward_decay': reward_decay})

        if clutter_density is not None:
            self.n_clutter = int(clutter_density * (self.width-2)*(self.height-2))
        else:
            self.n_clutter = n_clutter
        
        self.reward = reward
        self.reward_streak_bonus = reward_streak_bonus
        self.penalty = penalty

        self.initial_reward = initial_reward
        self.n_bonus_tiles = n_bonus_tiles
        self.reset_on_mistake = reset_on_mistake

        self.bonus_tiles = []

    def get_goal_location(self, agent):
        if not hasattr(self, 'bonus_tiles'):
            return np.array([0,0])
        if agent.bonus_state is None:
            tgt_bonus_id = 0
        else:
            tgt_bonus_id = agent.bonus_state[0] + 1
        return self.bonus_tiles[tgt_bonus_id%self.n_bonus_tiles].pos

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)

        for bonus_id in range(getattr(self, 'n_bonus_tiles', 0)):
            bonus_tile = BonusTile(
                color="yellow",
                reward=self.reward,
                penalty=self.penalty,
                reward_streak_bonus=self.reward_streak_bonus,
                bonus_id=bonus_id,
                n_bonus=self.n_bonus_tiles,
                initial_reward=self.initial_reward,
                reset_on_mistake=self.reset_on_mistake,
            )
            self.place_obj(bonus_tile, max_tries=100)
            self.bonus_tiles.append(bonus_tile)

        for _ in range(getattr(self, 'n_clutter', 0)):
            self.place_obj(Wall(), max_tries=100)

        self.agent_spawn_kwargs = {}
        self.place_agents(**self.agent_spawn_kwargs)
