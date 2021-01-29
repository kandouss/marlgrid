from ..base import MultiGridEnv, MultiGrid
from ..objects import *
import labmaze

class Maze(MultiGridEnv):
    mission = "get to the green square"
    metadata = {}

    def __init__(self, *args,  goal_color='green', **kwargs):
        self.goal_color = 'green'
        super().__init__(*args, **kwargs)


        self.goal_color = goal_color
        self.spawnpoint = (1,1)


        # self.reset()

    def _gen_maze(self, width, height):
        self._maze = labmaze.RandomMaze(
            height=((height-1)//2)*2+1, 
            width=((width-1)//2)*2+1,
            # max_rooms=5,
            simplify=False,
            extra_connection_probability=0.05)#0.1)#25)
        # import pdb; pdb.set_trace()
        self._maze_goals = labmaze.FixedMazeWithRandomGoals(
            entity_layer='\n'.join(''.join(row) for row in self._maze.entity_layer)+'\n',
            num_spawns=5,
            spawn_token='A',
            num_objects=1,
            object_token='G',
        )
        self.spawnpoints = []
        # print(self._maze_goals.entity_layer)
        spawnpoints = []
        for i, row in enumerate(self._maze.entity_layer):
            for j, el in enumerate(row):
                item = self._maze_goals.entity_layer[i][j]
                if el == '*':
                    self.grid.set(i, j, Wall())
                elif item == 'G':
                    self.grid.set(i, j, Goal(reward=1, color=self.goal_color))
                    goal_pos = (i,j)
                elif item == 'A':
                    spawnpoints.append((i,j))

        # Set the spawnpoint to be the farthest candidate spawn point from the goal.
        self.spawnpoint = max(
            spawnpoints,
            key=lambda p: abs(p[0]-goal_pos[0])+abs(p[1]-goal_pos[1])
        )
            

    def _gen_grid(self, width, height):
        self.grid = MultiGrid((width, height))
        self.grid.wall_rect(0, 0, width, height)
        self._gen_maze(width, height)

        sx, sy = self.spawnpoint
        self.agent_spawn_kwargs = {
            'top': (sx-1,sy-1),
            'size': (3, 3)
        }
        # self.place_agents(**self.agent_spawn_kwargs)
