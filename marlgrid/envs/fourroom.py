from ..base import MultiGridEnv, MultiGrid
from ..objects import *


class FourRoom(MultiGridEnv):
    mission = "get to the green square"
    metadata = {}

    def __init__(self, *args, n_clutter=None, clutter_density=None, randomize_goal=True, **kwargs):
        if (n_clutter is None) == (clutter_density is None):
            raise ValueError("Must provide n_clutter xor clutter_density in environment config.")

        super().__init__(*args, **kwargs)

        if clutter_density is not None:
            self.n_clutter = int(clutter_density * (self.width-2)*(self.height-2))
        else:
            self.n_clutter = n_clutter

        self.randomize_goal = randomize_goal


        # self.reset()


    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)

        self.wall_x = width//2
        self.wall_y = 4
        self.grid.horz_wall(1, self.height//2, length=self.width-2)
        self.grid.vert_wall(self.width//2, 1, length=self.height-2)

        if getattr(self, 'randomize_goal', True):
            candidates = [
                [(x+1,y+1), (x+1, y-1), (x-1,y+1), (x-1,y-1)]
                for x in [0, self.width//2, self.width-1]
                for y in [0, self.height//2, self.height-1]
            ]
            candidates = [
                (x,y) for p in candidates for (x,y) in p 
                if (1<x<self.width-1) and (1<y<self.height-1)
            ]
            goal_pos = candidates[self.np_random.randint(0, len(candidates))]
            self.put_obj(Goal(color="green", reward=1), *goal_pos)
        else:
            self.put_obj(Goal(color="green", reward=1), width - 2, height - 2)
        for _ in range(getattr(self, 'n_clutter', 0)):
            self.place_obj(Wall(), max_tries=100)

        self.grid.set(self.width//4, self.height//2, None)
        self.grid.set(self.width//2, self.height//4, None)
        self.grid.set((3*self.width-1)//4, self.height//2, None)
        self.grid.set(self.width//2, (3*self.height-1)//4, None)
        # self.grid.set(self.width//2, (3*self.height-1)//4, None)
        # self.grid.set((3*self.width-1)//4, (3*self.height-1)//4, None)

        self.agent_spawn_kwargs = {
            'top': (1,1),
            'size': (self.width//4-1, self.height//4-1)
        }
        self.place_agents(**self.agent_spawn_kwargs)
