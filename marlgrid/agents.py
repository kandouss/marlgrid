import gym
import numpy as np
from enum import IntEnum
import warnings
import numba

from .objects import GridAgent, BonusTile

class GridAgentInterface(GridAgent):
    class actions(IntEnum):
        left = 0  # Rotate left
        right = 1  # Rotate right
        forward = 2  # Move forward
        pickup = 3  # Pick up an object
        drop = 4  # Drop an object
        toggle = 5  # Toggle/activate an object
        done = 6  # Done completing task

    def __init__(
            self,
            view_size=7,
            view_tile_size=5,
            view_offset=0,
            observation_style='image',
            observe_rewards=False,
            observe_position=False,
            observe_orientation=False,
            restrict_actions=False,
            see_through_walls=False,
            hide_item_types=[],
            prestige_beta=0.95,
            prestige_scale=2,
            allow_negative_prestige=False,
            spawn_delay=0,
            **kwargs):
        super().__init__(**kwargs)

        self.view_size = view_size
        self.view_tile_size = view_tile_size
        self.view_offset = view_offset
        self.observation_style = observation_style
        self.observe_rewards = observe_rewards
        self.observe_position = observe_position
        self.observe_orientation = observe_orientation
        self.hide_item_types = hide_item_types
        self.see_through_walls = see_through_walls
        self.init_kwargs = kwargs
        self.restrict_actions = restrict_actions
        self.prestige_beta = prestige_beta
        self.prestige_scale = prestige_scale
        self.allow_negative_prestige = allow_negative_prestige
        self.spawn_delay = spawn_delay

        if self.prestige_beta > 1:
            # warnings.warn("prestige_beta must be between 0 and 1. Using default 0.99")
            self.prestige_beta = 0.95
            
        image_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(view_tile_size * view_size, view_tile_size * view_size, 3),
            dtype="uint8",
        )
        if observation_style == 'image':
            self.observation_space = image_space
        elif observation_style == 'rich':
            obs_space = {
                'pov': image_space,
            }
            if self.observe_rewards:
                obs_space['reward'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)
            if self.observe_position:
                obs_space['position'] = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
            if self.observe_orientation:
                obs_space['orientation'] = gym.spaces.Discrete(n=4)
            self.observation_space = gym.spaces.Dict(obs_space)
        else:
            raise ValueError(f"{self.__class__.__name__} kwarg 'observation_style' must be one of 'image', 'rich'.")

        if self.restrict_actions:
            self.action_space = gym.spaces.Discrete(3)
        else:
            self.action_space = gym.spaces.Discrete(len(self.actions))

        self.metadata = {
            **self.metadata,
            'view_size': view_size,
            'view_tile_size': view_tile_size,
        }
        self.reset(new_episode=True)

    def render_post(self, tile):
        if not self.active:
            return tile

        blue = np.array([0,0,255])
        red = np.array([255,0,0])

        if self.color == 'prestige':
            # Compute a scaled prestige value between 0 and 1 that will be used to 
            #   interpolate between the low-prestige (red) and high-prestige (blue)
            #   colors.
            if self.allow_negative_prestige:
                prestige_scaled = 1/(1 + np.exp(-self.prestige/self.prestige_scale))
            else:
                prestige_scaled = np.tanh(self.prestige/self.prestige_scale)

            new_color = (
                    prestige_scaled * blue +
                    (1.-prestige_scaled) * red
                ).astype(np.int)

            grey_pixels = (np.diff(tile, axis=-1)==0).all(axis=-1)

            alpha = tile[...,0].astype(np.uint16)[...,None]
            tile = np.right_shift(alpha * new_color, 8).astype(np.uint8)
            return tile
        else:
            return tile

    def clone(self):
        ret =  self.__class__(
            view_size = self.view_size,
            view_offset=self.view_offset,
            view_tile_size = self.view_tile_size,
            observation_style = self.observation_style,
            observe_rewards = self.observe_rewards,
            observe_position = self.observe_position,
            observe_orientation = self.observe_orientation,
            hide_item_types = self.hide_item_types,
            restrict_actions = self.restrict_actions,
            see_through_walls=self.see_through_walls,
            prestige_beta = self.prestige_beta,
            prestige_scale = self.prestige_scale,
            allow_negative_prestige = self.allow_negative_prestige,
            spawn_delay = self.spawn_delay,
            **self.init_kwargs
        )
        return ret

    def on_step(self, obj):
        if isinstance(obj, BonusTile):
            self.bonuses.append((obj.bonus_id, self.prestige))
        self.prestige *= self.prestige_beta

    def reward(self, rew):
        if self.allow_negative_prestige:
            self.rew += rew
        else:
            if rew >= 0:
                self.prestige += rew
            else: # rew < 0
                self.prestige = 0

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def reset(self, new_episode=False):
        self.done = False
        self.active = False
        self.pos = None
        self.carrying = None
        self.mission = ""
        if new_episode:
            self.prestige = 0
            self.bonus_state = None
            self.bonuses = []

    def render(self, img):
        if self.active:
            super().render(img)

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        assert self.dir >= 0 and self.dir < 4
        return np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])[self.dir]

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

        
        ax -= 2*self.view_offset*dx
        ay -= 2*self.view_offset*dy


        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

        
    def get_view_pos(self):
        return (self.view_size // 2, self.view_size - 1 - self.view_offset)


    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        dir = self.dir
        # Facing right
        if dir == 0:  # 1
            topX = self.pos[0] - self.view_offset
            topY = self.pos[1] - self.view_size // 2
        # Facing down
        elif dir == 1:  # 0
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_offset
        # Facing left
        elif dir == 2:  # 3
            topX = self.pos[0] - self.view_size + 1 + self.view_offset
            topY = self.pos[1] - self.view_size // 2
        # Facing up
        elif dir == 3:  # 2
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1 + self.view_offset
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

    def process_vis(self, opacity_grid):
        assert len(opacity_grid.shape) == 2
        if not self.see_through_walls:
            return occlude_mask(~opacity_grid, self.get_view_pos())
        else:
            return np.full(opacity_grid.shape, 1, dtype=np.bool)
    

@numba.njit
def occlude_mask(grid, agent_pos):
    mask = np.zeros(grid.shape[:2]).astype(numba.boolean)
    mask[agent_pos[0], agent_pos[1]] = True
    width, height = grid.shape[:2]

    for j in range(agent_pos[1]+1,0,-1):
        for i in range(agent_pos[0], width):
            if mask[i,j] and grid[i,j]:
                if i < width - 1:
                    mask[i + 1, j] = True
                if j > 0:
                    mask[i, j - 1] = True
                    if i < width - 1:
                        mask[i + 1, j - 1] = True

        for i in range(agent_pos[0]+1,0,-1):
            if mask[i,j] and grid[i,j]:    
                if i > 0:
                    mask[i - 1, j] = True
                if j > 0:
                    mask[i, j - 1] = True
                    if i > 0:
                        mask[i - 1, j - 1] = True


    for j in range(agent_pos[1], height):
        for i in range(agent_pos[0], width):
            if mask[i,j] and grid[i,j]:
                if i < width - 1:
                    mask[i + 1, j] = True
                if j < height-1:
                    mask[i, j + 1] = True
                    if i < width - 1:
                        mask[i + 1, j + 1] = True

        for i in range(agent_pos[0]+1,0,-1):
            if mask[i,j] and grid[i,j]:
                if i > 0:
                    mask[i - 1, j] = True
                if j < height-1:
                    mask[i, j + 1] = True
                    if i > 0:
                        mask[i - 1, j + 1] = True
                    
    return mask