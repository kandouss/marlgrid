'''
Multi-agent gridworld.
Largely a refactor of gym_minigrid.minigrid.
'''

import gym
import numpy as np
import gym_minigrid
from gym_minigrid.rendering import *
from enum import IntEnum
import math
TILE_PIXELS = 32


# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
    'worst' : np.array([74, 65, 42]), # https://en.wikipedia.org/wiki/Pantone_448_C
}
# Used to map colors to integers
COLOR_TO_IDX = dict({v:k for k,v in enumerate(COLORS.keys())})

OBJECT_TYPE_REGISTRY = []

class MetaRegistry(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        if name not in OBJECT_TYPE_REGISTRY:
            OBJECT_TYPE_REGISTRY.append(cls)
        def get_recursive_subclasses(x):
            return OBJECT_TYPE_REGISTRY
        cls.recursive_subclasses = get_recursive_subclasses
        return cls

class WorldObj(metaclass=MetaRegistry):
    def __init__(self,  color='worst', state=0):
        self.color = color
        self.state = state
        self.contains = None
        
        self.agent = None # Some objects can have agents on top (e.g. floor, open doors, etc).

        self.pos_init = None
        self.pos = None

    @property
    def dir(self):
        return None

    @property
    def type(self):
        return self.__class__.__name__

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False

    def can_contain(self):
        return False

    def see_behind(self):
        return True

    def toggle(self, env, pos):
        return False

    def encode(self, str_class = False):
        if self.agent is not None:
            return self.agent.encode(str_class=str_class)
        else:
            if str_class:
                return (self.type, self.color, self.state)
            else:
                # return (WorldObj.__subclasses__().index(self.__class__), self.color, self.state)
                return (
                    self.recursive_subclasses().index(self.__class__), 
                    self.color if isinstance(self.color, int) else COLOR_TO_IDX[self.color],
                    self.state)

    def describe(self):
        return f'Obj: {self.type}({self.color}, {self.state})'

    @classmethod
    def decode(cls, type, color, state):
        if isinstance(type, str):
            cls_subclasses = {c.__name__: c for c in cls.__subclasses__()}
            if type not in cls_subclasses:
                raise ValueError(f"Not sure how to construct a {cls} of (sub)type {type}")
            return cls_subclasses[type](color, state)
        elif isinstance(type, int):
            subclass = cls.__subclasses__()[type]
            return subclass(color, state)

    
    def render(self, img):
        raise NotImplementedError

    def str_render(self, dir=0):
        return '??'

class BulkObj(WorldObj):
    def __hash__(self):
        return hash((self.__class__, self.color, self.state, self.agent))
    def __eq__(self, other):
        return hash(self)==hash(other)

class Goal(WorldObj):
    def __init__(self, reward, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = reward

    def can_overlap(self):
        return True

    def str_render(self, dir=0):
        return 'GG'

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Agent(WorldObj):
    @property
    def dir(self):
        return self.state%4

    @dir.setter
    def dir(self, dir):
        self.state = self.state // 4 + dir%4

    def str_render(self, dir=0):
        return ['>>','VV','<<','^^'][(self.dir+dir)%4]

    @property
    def active(self):
        return False

    def render(self, img):
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*(self.dir))
        fill_coords(img, tri_fn, COLORS[self.color])
    
class Floor(WorldObj):
    def can_overlap(self):
        return True and self.agent is None
    def str_render(self, dir=0):
        return 'FF'    
    def render(self, img):
        # Give the floor a pale color
        c = COLORS[self.color]
        r.setLineColor(100, 100, 100, 0)
        r.setColor(*c/2)
        r.drawPolygon([
            (1          , TILE_PIXELS),
            (TILE_PIXELS, TILE_PIXELS),
            (TILE_PIXELS,           1),
            (1          ,           1)
        ])

class EmptySpace(WorldObj):
    def can_verlap(self):
        return True
    def str_render(self, dir=0):
        return '  '

class Lava(WorldObj):
    def can_overlap(self):
        return True and self.agent is None
    def str_render(self, dir=0):
        return 'VV'
    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0,0,0))
        
    

class Wall(BulkObj):
    def see_behind(self):
        return False
    def str_render(self, dir=0):
        return 'WW'
    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Key(WorldObj):
    def can_pickup(self):
        return True
    def str_render(self, dir=0):
        return 'KK'
    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0,0,0))

class Ball(WorldObj):
    def can_pickup(self):
        return True
    def str_render(self, dir=0):
        return 'AA'

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Door(WorldObj):
    states = IntEnum('door_state','open closed locked')
    def can_overlap(self):
        return self.state == self.states.open  and self.agent is None # is open
    def see_behind(self):
        return self.state == self.states.open # is open
    def toggle(self, agent, pos):
        if self.state == self.states.locked: # is locked
            # If the agent is carrying a key of matching color
            if agent.carrying is not None and isinstance(agent.carrying, Key) and agent.carrying.color == self.color:
                self.state = self.states.closed
        elif self.state == self.states.closed: # is unlocked but closed
            self.state = self.states.open
        elif self.state == self.states.open: # is open
            self.state = self.states.closed
        return True    


    def render(self, img):
        c = COLORS[self.color]

        if self.state==self.states.open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0,0,0))
            return

        # Door frame and door
        if self.state==self.states.locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0,0,0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0,0,0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c) 

    
class Box(WorldObj):
    def __init__(self, color=0, state=0, contains=None):
        super().__init__(color, state)
        self.contains = contains

    def can_pickup(self):
        return True

    def toggle(self):
        raise NotImplementedError

    def str_render(self, dir=0):
        return 'BB'

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)
    
class ObjectRegistry:
    def __init__(self, objs=[], max_num_objects = 1000):
        self.key_to_obj_map = {}
        self.obj_to_key_map = {}
        self.max_num_objects = max_num_objects
        for obj in objs:
            self.add_object(obj)

    def get_next_key(self):
        for k in range(self.max_num_objects):
            if k not in self.key_to_obj_map:
                break
        else:
            raise ValueError("Object registry full.")
        return k

    def __len__(self):
        return len(self.id_to_obj_map)

    def add_object(self, obj):
        new_key = self.get_next_key()
        self.key_to_obj_map[new_key] = obj
        self.obj_to_key_map[obj] = new_key
        return new_key

    def contains_object(self, obj):
        return obj in self.obj_to_key_map
    
    def contains_key(self, key):
        return key in self.key_to_obj_map

    def get_key(self, obj):
        if obj in self.obj_to_key_map:
            return self.obj_to_key_map[obj]
        else:
            return self.add_object(obj)
    
    def obj_of_key(self, key):
        return self.key_to_obj_map[key]

class MultiGrid:

    tile_cache = {}

    def __init__(self, shape, obj_reg=None, orientation=0):
        self.orientation=orientation
        if isinstance(shape, tuple):
            self.width, self.height = shape
            self.grid = np.zeros((self.width, self.height), dtype=np.uint8) # w,h
        elif isinstance(shape, np.ndarray):
            self.width, self.height = shape.shape
            self.grid = shape
        else:
            # print(shape)
            raise ValueError("Must create grid from shape tuple or array.")

        if self.width < 3 or self.height < 3:
            raise ValueError("Grid needs width, height >= 3")

        self.obj_reg = ObjectRegistry(objs=[None]) if obj_reg is None else obj_reg

    def __getitem__(self, *args, **kwargs):
        return self.__class__(np.ndarray.__getitem__(self.grid, *args, **kwargs), obj_reg=self.obj_reg, orientation=self.orientation)

    def rotate_left(self, k=1):
        return self.__class__(np.rot90(self.grid,k=k), obj_reg=self.obj_reg, orientation=(self.orientation-k)%4)

    def slice(self, topX, topY, width, height, rot_k=0):
        """
        Get a subset of the grid
        """
        sub_grid = self.__class__((width, height), obj_reg=self.obj_reg, orientation=(self.orientation-rot_k)%4)
        x_min = max(0, topX)
        x_max = min(topX+width, self.width)
        y_min = max(0, topY)
        y_max = min(topY+height, self.height)

        x_offset = x_min - topX
        y_offset = y_min - topY
        sub_grid.grid[x_offset:x_max-x_min+x_offset, y_offset:y_max-y_min+y_offset] = self.grid[x_min:x_max, y_min:y_max]
        sub_grid.grid = np.rot90(sub_grid.grid, k=-rot_k)
        sub_grid.width, sub_grid.height = sub_grid.grid.shape        

        return sub_grid


    def set(self, i, j, obj):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[i,j] = self.obj_reg.get_key(obj)

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height

        return self.obj_reg.obj_of_key(self.grid[i,j])

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x+i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def __str__(self):
        render = lambda x: '  ' if x is None or not hasattr(x, 'str_render') else x.str_render(dir=self.orientation)
        hstars = '*'*(2*self.width + 2)
        return (
             hstars +'\n'+
            '\n'.join(
                '*'+''.join(render(self.get(i,j)) for i in range(self.width) )+'*'
             for j in range(self.height)
        )+'\n'+hstars)

    def encode(self, vis_mask = None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)
                    if v is None:
                        array[i, j, :] = 0
                    else:
                        array[i, j, :] = v.encode()
        return array

    @classmethod
    def decode(cls,array):
        raise NotImplementedError
        width, height, channels = array.shape
        assert channels == 3
        # objects = {k: WorldObj.decode(k) for k in np.unique(array[:,:,0])}
        # print(objects)
        vis_mask[i,j] = np.ones(shape=(width, height), dtype=np.bool)
        grid = cls((width, height))


    def process_vis(grid, agent_pos):
        mask = np.zeros_like(grid.grid, dtype=np.bool)
        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width-1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i+1, j] = True
                if j > 0:
                    mask[i+1, j-1] = True
                    mask[i, j-1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i-1, j] = True
                if j > 0:
                    mask[i-1, j-1] = True
                    mask[i, j-1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask

    @classmethod
    def render_tile(cls, obj, highlight=False, tile_size=TILE_PIXELS, subdivs=3):
        if obj is None:
            key = (tile_size, highlight, )
        else:
            key = (tile_size, highlight, *obj.encode())

        if key in cls.tile_cache:
            img = cls.tile_cache[key]
        else:
            img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

            # Draw the grid lines (top and left edges)
            fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
            fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

            if obj != None:
                obj.render(img)
            
            if highlight:
                highlight_img(img)

            img = downsample(img, subdivs)

            cls.tile_cache[key] = img

        return img

    def render(self, tile_size, highlight_mask=None):

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)


        for j in range(0, self.height):
            for i in range(0, self.width):
                obj = self.get(i,j)
                

                tile_img = MultiGrid.render_tile(
                    obj,
                    highlight=highlight_mask[i,j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] =  np.rot90(tile_img, -self.orientation)

        return img


class GridAgent(Agent):
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4
        # Toggle/activate an object
        toggle = 5

        # Done completing task
        done = 6

    def __init__(self, view_size, view_tile_size=7, actions=None, **kwargs):
        super().__init__(**{'color':'red', **kwargs})
        if actions is None:
            actions = GridAgent.Actions

        self.actions = actions
        self.view_size = view_size
        self.view_tile_size = view_tile_size


        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(view_tile_size*view_size, view_tile_size*view_size, 3), dtype='uint8')

        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.done = True

        self.reset()

    def reset(self):
        self.done = False
        self.pos = None
        self.dir = 0
        self.carrying = None
        self.mission = ''

    def render(self, img):
        if not self.done:
            super().render(img)

    @property
    def active(self):
        return not self.done

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        # print(f"DIR IS {self.dir}")
        assert self.dir >= 0 and self.dir < 4
        return np.array([[1,0],[0,1],[-1,0],[0,-1]])[self.dir]
    
    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """
        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """
        return np.add(self.pos, self.dir_vec)

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz-1)) - (rx * hs)
        ty = ay + (dy * (sz-1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx*lx + ry*ly)
        vy = -(dx*lx + dy*ly)

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """
        dir = self.dir
        # Facing right
        if dir == 0: # 1
            topX = self.pos[0]
            topY = self.pos[1] - self.view_size // 2
        # Facing down
        elif dir == 1: # 0
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1]
        # Facing left
        elif dir == 2: # 3
            topX = self.pos[0] - self.view_size + 1
            topY = self.pos[1] - self.view_size // 2
        # Facing up
        elif dir == 3: # 2
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def sees(self, x, y):
        raise NotImplementedError


class MultiGridEnv:

    def __init__(self, agents, grid_size=None, width=None, height=None, max_steps=100, see_through_walls=False, seed=1337):


        if grid_size is not None:
            assert width == None and height == None
            width, height = grid_size, grid_size

        
        self.num_agents = len(agents)
        self.agents = agents

        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(len(agent.actions)) for agent in self.agents))
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=255, shape=(agent.view_size, agent.view_size, 3), dtype='uint8')
            for agent in self.agents)
        )
        self.reward_range = [(0,1) for _ in range(len(self.agents))]

        self.window = None

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        self.seed(seed=seed)

        self.reset()

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def reset(self):
        for agent in self.agents:
            agent.reset()

        self._gen_grid(self.width, self.height)

        for agent in self.agents:
            # Make sure _gen_grid initialized agent positions
            assert (agent.pos is not None) and (agent.dir is not None)
            # Make sure the agent doesn't overlap with an object
            start_cell = self.grid.get(*agent.pos)
            # assert start_cell is None or start_cell.can_overlap()
            assert start_cell is agent

        self.step_count = 0

        obs = self.gen_obs()
        return obs

    def gen_obs_grid(self, agent):
        topX, topY, botX, botY = agent.get_view_exts()

        grid = self.grid.slice(topX, topY, agent.view_size, agent.view_size, rot_k = agent.dir + 1)

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(agent.view_size // 2 , agent.view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)
        
        return grid, vis_mask

    def gen_agent_obs(self, agent):
        grid, vis_mask = self.gen_obs_grid(agent)
        return grid.render(tile_size=agent.view_tile_size)#,highlight_mask=~vis_mask)
    
    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        # obs_list = []
        # for agent in self.agents:
        #     grid, vis_mask = self.gen_obs_grid(agent)

        #     obs_list.append({
        #         'image': grid.encode(vis_mask),
        #         'direction': agent.dir,
        #         'mission': agent.mission
        #     })

        return  [self.gen_agent_obs(agent) for agent in self.agents]
        # return obs_list

    # def get_obs_render(self, obs, agent, tile_size=TILE_PIXELS//2):
    #     grid, vis_mask = MultiGrid.decode(obs)

    def __str__(self):
        return self.grid.__str__()

    def step(self, actions):
        assert len(actions) == len(self.agents)
        rewards = np.zeros((len(self.agents,)),dtype=np.float)

        self.step_count += 1

        wasteds = []

        for agent_no, (agent, action) in enumerate(zip(self.agents, actions)):
            wasted = False
            if agent.active:
                
                cur_pos = agent.pos
                cur_cell = self.grid.get(*cur_pos)
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)

                # Rotate left
                if action == GridAgent.Actions.left:
                    agent.dir = (agent.dir - 1)%4

                # Rotate right
                elif action == GridAgent.Actions.right:
                    agent.dir = (agent.dir + 1)%4

                # Move forward
                elif action == GridAgent.Actions.forward:
                    # Under these conditions, the agent can move forward.
                    if (fwd_cell is None) or fwd_cell.can_overlap():

                        # Move the agent to the forward cell
                        agent.pos = fwd_pos

                        if fwd_cell is None:
                            self.grid.set(*fwd_pos, agent)
                        elif fwd_cell.can_overlap():
                            fwd_cell.agent = agent

                        
                        if not isinstance(cur_cell, GridAgent): 
                            cur_cell.agent = None
                        else:
                            self.grid.set(*cur_pos, None)

                    else:
                        wasted = True


                    if isinstance(fwd_cell, Goal): # No extra wasting logic
                        rewards[agent_no] += fwd_cell.reward
                        agent.done = True
                        fwd_cell.agent = None

                    if isinstance(fwd_cell, Lava):
                        agent.done = True

                # Pick up an object
                elif action == GridAgent.Actions.pickup:
                    if fwd_cell and fwd_cell.can_pickup():
                        if agent.carrying is None:
                            agent.carrying = fwd_cell
                            agent.carrying.cur_pos = np.array([-1, -1])
                            self.grid.set(*fwd_pos, None)
                    else:
                        wasted = True

                # Drop an object
                elif action == GridAgent.Actions.drop:
                    if not fwd_cell and agent.carrying:
                        self.grid.set(*fwd_pos, agent.carrying)
                        agent.carrying.cur_pos = fwd_pos
                        agent.carrying = None
                    else:
                        wasted = True


                # Toggle/activate an object
                elif action == GridAgent.Actions.toggle:
                    if fwd_cell:
                        wasted = bool(fwd_cell.toggle(agent, fwd_pos))
                    else:
                        wasted = True
                    

                # Done action (not used by default)
                elif action == GridAgent.Actions.done:
                    # dones[agent_no] = True
                    wasted = True

                else:
                    raise ValueError(f"Environment can't handle action {action}.")
            wasteds.append(wasted)
                
        dones = np.array([agent.done for agent in self.agents], dtype=np.bool)
        if self.step_count >= self.max_steps:
            dones[:] = True

        obs = [self.gen_agent_obs(agent) for agent in self.agents]

        wasteds = np.array(wasteds, dtype=np.bool)

        return obs, rewards, dones, wasteds


    @property
    def agent_positions(self):
        return [tuple(agent.pos) if agent.pos is not None else None for agent in self.agents]
    
    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        max_tries = int(max(1, min(max_tries, 1e5)))
        if top is None:
            top = (0,0)
        else:
            top = (max(top[0], 0), max(top[1], 0))
        if size is None:
            size = (self.grid.width, self.grid.height)

        agent_positions = self.agent_positions
        for try_no in range(max_tries):
            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            )

            if((self.grid.get(*pos) is None)
                and (pos not in agent_positions)
                and (reject_fn is None or (not reject_fn(pos)))
                ): break
        else:
            raise RecursionError("Rejection sampling failed in place_obj.")

        self.grid.set(*pos,obj)
        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(self, agent, top=None, size=None, rand_dir=True, max_tries=100):
        agent.pos = self.place_obj(agent, top=top, size=size, max_tries=max_tries)
        if rand_dir:
            agent.dir = self._rand_int(0,4)
        return agent

    def place_agents(self, top=None, size=None, rand_dir=True, max_tries=100):
        for agent in self.agents:
            self.place_agent(agent, top=top, size=size, rand_dir=rand_dir, max_tries=max_tries)
            if hasattr(self, 'mission'):
                agent.mission = self.mission

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS, show_agent_views=True, max_agents_per_col=3,):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            from gym.envs.classic_control.rendering import SimpleImageViewer
            self.window = SimpleImageViewer()
            # self.window.show(block=False)

        # Compute which cells are visible to the agent
        highlight_mask = np.full((self.width, self.height), False, dtype=np.bool)
        for agent in self.agents:
            xlow, ylow, xhigh, yhigh = agent.get_view_exts()
            if agent.active:
                highlight_mask[
                    max(0,xlow):min(self.grid.width,xhigh),
                    max(0,ylow):min(self.grid.height,yhigh)] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            highlight_mask=highlight_mask if highlight else None
        )
        rescale = lambda X, rescale_factor=2: np.kron(X, np.ones((rescale_factor, rescale_factor, 1)))
        
        if show_agent_views:
            agent_no = 0
            cols = []
            rescale_factor = None

            for col_no in range(len(self.agents)//(max_agents_per_col+1)+1):
                col_count = min(max_agents_per_col, len(self.agents)-agent_no)
                views = []
                for row_no in range(col_count):
                    tmp = self.gen_agent_obs(self.agents[agent_no])
                    if rescale_factor is None:
                        rescale_factor = img.shape[0]//(min(3,col_count)*tmp.shape[1])
                    views.append(rescale(tmp, rescale_factor))
                    agent_no += 1

                col_width = max([v.shape[1] for v in views])
                img_col = np.zeros((img.shape[0], col_width, 3), dtype=np.uint8)
                for k,view in enumerate(views):
                    start_x = (k*img.shape[0])//len(views)
                    start_y = 0#(k*img.shape[1])//len(views)
                    dx, dy = view.shape[:2]
                    img_col[start_x:start_x+dx, start_y:start_y+dy,:] = view
                cols.append(img_col)
            img = np.concatenate((img, *cols),axis=1)

        if mode == 'human':
            self.window.imshow(img)

        return img
