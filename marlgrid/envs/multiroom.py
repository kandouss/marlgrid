from ..base import MultiGridEnv, MultiGrid
from ..objects import *

## 
#   Follows MiniGrid's multiroom implementation
#   https://github.com/maximecb/gym-minigrid/blob/master/gym_minigrid/envs/multiroom.py
##



class Room:
    def __init__(self,
        top,
        size,
        entry_pos,
        exit_pos,
        entry_wall,
        exit_wall
    ):
        self.top = top
        self.size = size
        self.entry_pos = entry_pos
        self.exit_pos = exit_pos
        self.entry_wall = entry_wall
        self.exit_wall = exit_wall

    def translate(self, dx, dy):
        self.top = (self.top[0] + dx, self.top[1] + dy)
        self.entry_pos = (self.entry_pos[0] + dx, self.entry_pos[1] + dy)
        self.exit_pos = (self.exit_pos[0] + dx, self.exit_pos[1] + dy)

class MultiRoomEnv(MultiGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    DOORCOLORS = ["red", "orange", "green", "blue", "cyan", "purple", "yellow"]

    def __init__(self,
        *args,
        min_rooms=2,
        max_rooms=6,
        min_room_size=5,
        max_room_size=8,
        doors=False,
        goal_color='green',
        grid_size=30,
        max_tries=1e4,
        **kwargs
    ):
        assert min_rooms > 0
        assert max_rooms >= min_rooms
        assert max_room_size >= 4

        self.doors = doors
        self.min_rooms = min_rooms
        self.max_rooms = max_rooms
        self.min_room_size = min_room_size
        self.max_room_size = max_room_size
        self.goal_color = goal_color
        self.max_tries = max_tries
        self.n_tries = 0
        

        self.rooms = []

        super().__init__(
            *args,
            grid_size=grid_size,
            **kwargs,
        )

    def grid_limits(self):
        x_min = min(room.top[0] for room in self.rooms)
        y_min = min(room.top[1] for room in self.rooms)
        x_max = max(room.top[0]+room.size[0]+2 for room in self.rooms)
        y_max = max(room.top[1]+room.size[1]+2 for room in self.rooms)
        return (x_min, x_max), (y_min, y_max)


    def _gen_grid(self, width, height):
        n_rooms = self.np_random.randint(self.min_rooms, self.max_rooms+1)
        self.rooms = []

        self.n_tries = 0
        while len(self.rooms) < n_rooms:
            self.n_tries += 1
            assert self.max_tries > self.n_tries, "Couldn't generate rooms. Try making grid size bigger or reducing room size."
            self.rooms = []
            for room_num in range(n_rooms):
                if not self.place_room():
                    continue
            
            xb, yb = self.grid_limits()
            if ((yb[1]-yb[0])<self.height) and ((xb[1]-xb[0])<self.width):
                for room in self.rooms:
                    room.translate(-xb[0]+1,-yb[0]+1)
            else:
                self.rooms = []
            
                    

        self.grid = MultiGrid((width, height))

        # self.grid.wall_rect((0,0), (xb[1]-xb[0], yb[1]-yb[0]))

        for ix, room in enumerate(self.rooms):
            # Draw the top and bottom walls
            self.grid.wall_rect(*room.top, *room.size, obj_type=Wall)

            if ix==0:
                door_colors = [0]
            else:
                door_colors.append(self._randint_except(len(self.DOORCOLORS), exclude=door_colors[-1]))

                if self.doors:
                    # pick a new door color (different from the previous one)
                    self.grid.set(*room.entry_pos, Door(color=self.DOORCOLORS[door_colors[-1]]))
                else:
                    self.grid.set(*room.entry_pos, None)

        t = self.rooms[-1].top
        s = self.rooms[-1].size
        self.goal_pos = self.place_obj(
            Goal(color=self.goal_color, reward=1),
             top=(t[0]+1,t[1]+1), size=(s[0]-2,s[1]-2))

        self.agent_spawn_kwargs = {
            'top': self.rooms[0].top,
            'size': self.rooms[0].size,
        }

    def _randint_except(self, low, high=None, exclude=None, max_tries=1000):
        for _ in range(max_tries):
            ret = self.np_random.randint(low, high)
            if ret != exclude:
                return ret
        assert False, "Exceeded max tries."
        
    def random_pt_on_wall(self, width, height, wall, exclude_corners=True):
        dh = int(bool(exclude_corners))
        randint = self.np_random.randint
        if wall == 0:
            return (width-1, randint(dh,height-dh))

        elif wall == 1:
            return (randint(dh,width-dh), height-1)

        elif wall == 2:
            return (0, randint(dh,height-dh))

        elif wall == 3:
            return (randint(dh,width-dh), 0)

        raise Exception("Wall should be one of [0,1,2,3]")

    def place_room(self):
        size = tuple(self.np_random.randint(self.min_room_size, self.max_room_size+1, 2))

        if len(self.rooms) == 0:
            entry_wall = 2
            entry_pos_abs = tuple(np.random.randint((self.width-2, self.height-2)))
            entry_pos_rel = (0,0)
        else:
            entry_wall = (self.rooms[-1].exit_wall+2)%4
            entry_pos_abs = self.rooms[-1].exit_pos
            entry_pos_rel = self.random_pt_on_wall(*size, entry_wall)

        top = (entry_pos_abs[0]-entry_pos_rel[0] , entry_pos_abs[1]-entry_pos_rel[1])

        exit_wall = self._randint_except(4, exclude=entry_wall)
        exit_pos_rel = self.random_pt_on_wall(*size, exit_wall)
        exit_pos_abs = (exit_pos_rel[0]+top[0] , exit_pos_rel[1]+top[1])

        # # Check that the room is entirely in the grid
        # if ( (top[0]<0) or (top[1]<0) or 
        #      (top[0]+size[0]>self.width) or (top[1]+size[1]>self.height) ):
        #     return False

        # Check that the room doesn't overlap with the previous rooms
        for room in self.rooms[:-1]:
            # if (
            #     top[0]+size[0] > room.top[0] or
            #     top[0] <= 
            # )
            if not (
                top[0]+size[0] < room.top[0] or
                top[0] > room.top[0]+room.size[0] or
                top[1]+size[1] < room.top[1] or
                top[1] > room.top[1]+room.size[1]
            ):
                return False
        
        self.rooms.append(Room(
            top=top,
            size=size,
            entry_pos=entry_pos_abs,
            exit_pos=exit_pos_abs,
            entry_wall=entry_wall,
            exit_wall=exit_wall
        ))
        return True
