from .base import (
    MultiGridEnv,
    MultiGrid,
    Goal,
    Wall,
    Door,
    Key
)

class EmptyMultiGrid(MultiGridEnv):
    mission='get to the green square'
    metadata = {}

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Goal(color='green', reward=1), width-2, height-2)

        self.place_agents()


class ClutteredMultiGrid(MultiGridEnv):
    mission='get to the green square'
    metadata = {}

    def __init__(self, *args, n_clutter=25, randomize_goal=False, **kwargs):
        self.n_clutter = n_clutter
        self.randomize_goal = randomize_goal
        super().__init__(*args, **kwargs)
    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        if self.randomize_goal:
            self.place_obj(Goal(color='green', reward=1), max_tries=100)
        else:
            self.put_obj(Goal(color='green', reward=1), width-2, height-2)
        for _ in range(self.n_clutter):
            self.place_obj(Wall(), max_tries=100)

        self.place_agents()


class VisibilityTestEnv(MultiGridEnv):
    mission='get to the green square'
    metadata={}

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        self.grid.horz_wall(0, height//2, width-3, obj_type=Wall)

        self.place_agents()

class DoorKeyEnv(MultiGridEnv):
    """
    Environment with a door and key, sparse reward.
    Similar to DoorKeyEnv in 
        https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/envs/doorkey.py
    """
    mission = "use the key to open the door and then get to the goal"
    metadata = {}

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MultiGrid((width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(color='green',reward=1), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width-2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        # self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door(color='yellow', state=Door.states.locked), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.place_agents()