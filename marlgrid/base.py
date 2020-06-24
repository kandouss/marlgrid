# Multi-agent gridworld.
# Based on MiniGrid: https://github.com/maximecb/gym-minigrid.

import gym
import numpy as np
import gym_minigrid
from enum import IntEnum
import math

from .objects import WorldObj, Wall, Goal, Lava, GridAgent, BonusTile, BulkObj, COLORS
from gym_minigrid.rendering import fill_coords, point_in_rect, downsample, highlight_img

TILE_PIXELS = 32


class ObjectRegistry:
    def __init__(self, objs=[], max_num_objects=1000):
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

    # 5/4/2020 This gets called A LOT. Replaced calls to this function with direct dict gets
    #           in an attempt to speed things up. Probably didn't make a big difference.
    def obj_of_key(self, key):
        return self.key_to_obj_map[key]

def rotate_grid(grid, rot_k):        
    rot_k = rot_k % 4
    if rot_k==3:
        return np.moveaxis(grid[:,::-1], 0, 1)
    elif rot_k==1:
        return np.moveaxis(grid[::-1,:], 0, 1)
    elif rot_k==2:
        return grid[::-1,::-1]
    else:
        return grid


class MultiGrid:

    tile_cache = {}

    def __init__(self, shape, obj_reg=None, orientation=0):
        self.orientation = orientation
        if isinstance(shape, tuple):
            self.width, self.height = shape
            self.grid = np.zeros((self.width, self.height), dtype=np.uint8)  # w,h
        elif isinstance(shape, np.ndarray):
            self.width, self.height = shape.shape
            self.grid = shape
        else:
            raise ValueError("Must create grid from shape tuple or array.")

        if self.width < 3 or self.height < 3:
            raise ValueError("Grid needs width, height >= 3")

        self.obj_reg = ObjectRegistry(objs=[None]) if obj_reg is None else obj_reg

    @property
    def opacity(self):
        transparent_fun = np.vectorize(lambda k: (self.obj_reg.key_to_obj_map[k].see_behind() if hasattr(self.obj_reg.key_to_obj_map[k], 'see_behind') else True))
        return ~transparent_fun(self.grid)

    def __getitem__(self, *args, **kwargs):
        return self.__class__(
            np.ndarray.__getitem__(self.grid, *args, **kwargs),
            obj_reg=self.obj_reg,
            orientation=self.orientation,
        )

    def rotate_left(self, k=1):
        return self.__class__(
            rotate_grid(self.grid, rot_k=k), # np.rot90(self.grid, k=k),
            obj_reg=self.obj_reg,
            orientation=(self.orientation - k) % 4,
        )


    def slice(self, topX, topY, width, height, rot_k=0):
        """
        Get a subset of the grid
        """
        sub_grid = self.__class__(
            (width, height),
            obj_reg=self.obj_reg,
            orientation=(self.orientation - rot_k) % 4,
        )
        x_min = max(0, topX)
        x_max = min(topX + width, self.width)
        y_min = max(0, topY)
        y_max = min(topY + height, self.height)

        x_offset = x_min - topX
        y_offset = y_min - topY
        sub_grid.grid[
            x_offset : x_max - x_min + x_offset, y_offset : y_max - y_min + y_offset
        ] = self.grid[x_min:x_max, y_min:y_max]

        sub_grid.grid = rotate_grid(sub_grid.grid, rot_k)

        sub_grid.width, sub_grid.height = sub_grid.grid.shape

        return sub_grid

    def set(self, i, j, obj):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[i, j] = self.obj_reg.get_key(obj)

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height

        return self.obj_reg.key_to_obj_map[self.grid[i, j]]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h, obj_type=Wall):
        self.horz_wall(x, y, w, obj_type=obj_type)
        self.horz_wall(x, y + h - 1, w, obj_type=obj_type)
        self.vert_wall(x, y, h, obj_type=obj_type)
        self.vert_wall(x + w - 1, y, h, obj_type=obj_type)

    def __str__(self):
        render = (
            lambda x: "  "
            if x is None or not hasattr(x, "str_render")
            else x.str_render(dir=self.orientation)
        )
        hstars = "*" * (2 * self.width + 2)
        return (
            hstars
            + "\n"
            + "\n".join(
                "*" + "".join(render(self.get(i, j)) for i in range(self.width)) + "*"
                for j in range(self.height)
            )
            + "\n"
            + hstars
        )

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype="uint8")

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
    def decode(cls, array):
        raise NotImplementedError
        width, height, channels = array.shape
        assert channels == 3
        vis_mask[i, j] = np.ones(shape=(width, height), dtype=np.bool)
        grid = cls((width, height))

    
    @classmethod
    def cache_render_fun(cls, key, f, *args, **kwargs):
        if key not in cls.tile_cache:
            cls.tile_cache[key] = f(*args, **kwargs)
        return np.copy(cls.tile_cache[key])

    @classmethod
    def cache_render_obj(cls, obj, tile_size, subdivs):
        if obj is None:
            return cls.cache_render_fun((tile_size, None), cls.empty_tile, tile_size, subdivs)
        else:
            img = cls.cache_render_fun(
                (tile_size, obj.__class__.__name__, *obj.encode()),
                cls.render_object, obj, tile_size, subdivs
            )
            if hasattr(obj, 'render_post'):
                return obj.render_post(img)
            else:
                return img

    @classmethod
    def empty_tile(cls, tile_size, subdivs):
        alpha = max(0, min(20, tile_size-10))
        img = np.full((tile_size, tile_size, 3), alpha, dtype=np.uint8)
        img[1:,:-1] = 0
        return img

    @classmethod
    def render_object(cls, obj, tile_size, subdivs):
        img = np.zeros((tile_size*subdivs,tile_size*subdivs, 3), dtype=np.uint8)
        obj.render(img)
        # if 'Agent' not in obj.type and len(obj.agents) > 0:
        #     obj.agents[0].render(img)
        return downsample(img, subdivs).astype(np.uint8)

    @classmethod
    def blend_tiles(cls, img1, img2):
        '''
        This function renders one "tile" on top of another. Kinda janky, works surprisingly well.
        Assumes img2 is a downscaled monochromatic with a black (0,0,0) background.
        '''
        alpha = img2.sum(2, keepdims=True)
        max_alpha = alpha.max()
        if max_alpha == 0:
            return img1
        return (
            ((img1 * (max_alpha-alpha))+(img2*alpha)
            )/max_alpha
        ).astype(img1.dtype)

    @classmethod
    def render_tile(cls, obj, tile_size=TILE_PIXELS, subdivs=3, top_agent=None):
        subdivs = 3

        if obj is None:
            img = cls.cache_render_obj(obj, tile_size, subdivs)
        else:
            if ('Agent' in obj.type) and (top_agent in obj.agents):
                # If the tile is a stack of agents that includes the top agent, then just render the top agent.
                img = cls.cache_render_obj(top_agent, tile_size, subdivs)
            else: 
                # Otherwise, render (+ downsize) the item in the tile.
                img = cls.cache_render_obj(obj, tile_size, subdivs)
                # If the base obj isn't an agent but has agents on top, render an agent and blend it in.
                if len(obj.agents)>0 and 'Agent' not in obj.type:
                    if top_agent in obj.agents:
                        img_agent = cls.cache_render_obj(top_agent, tile_size, subdivs)
                    else:
                        img_agent = cls.cache_render_obj(obj.agents[0], tile_size, subdivs)
                    img = cls.blend_tiles(img, img_agent)

            # Render the tile border if any of the corners are black.
            if (img[([0,0,-1,-1],[0,-1,0,-1])]==0).all(axis=-1).any():
                img = img + cls.cache_render_fun((tile_size, None), cls.empty_tile, tile_size, subdivs)
        return img

    def render(self, tile_size, highlight_mask=None, visible_mask=None, top_agent=None):
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px), dtype=np.uint8)[...,None]+COLORS['shadow']

        for j in range(0, self.height):
            for i in range(0, self.width):
                if visible_mask is not None and not visible_mask[i,j]:
                    continue
                obj = self.get(i, j)

                tile_img = MultiGrid.render_tile(
                    obj,
                    tile_size=tile_size,
                    top_agent=top_agent
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size

                img[ymin:ymax, xmin:xmax, :] = rotate_grid(tile_img, self.orientation)
        
        if highlight_mask is not None:
            hm = np.kron(highlight_mask.T, np.full((tile_size, tile_size), 255, dtype=np.uint16)
                )[...,None] # arcane magic.
            img = np.right_shift(img.astype(np.uint16)*8+hm*2, 3).clip(0,255).astype(np.uint8)

        return img


class MultiGridEnv(gym.Env):
    def __init__(
        self,
        agents,
        grid_size=None,
        width=None,
        height=None,
        max_steps=100,
        done_condition=None,
        reward_decay=True,
        seed=1337,
        respawn=False,
        ghost_mode=True,
        agent_spawn_kwargs = {}
    ):
        if grid_size is not None:
            assert width == None and height == None
            width, height = grid_size, grid_size

        if done_condition is not None and done_condition not in ("any", "all"):
            raise ValueError("done_condition must be one of ['any', 'all', None].")
        self.done_condition = done_condition

        self.num_agents = len(agents)
        self.agents = agents
        self.respawn = respawn

        self.action_space = gym.spaces.Tuple(
            [agent.action_space for agent in self.agents]
        )
        self.observation_space = gym.spaces.Tuple(
            [agent.observation_space for agent in self.agents]
        )
        self.reward_range = [(0, 1) for _ in range(len(self.agents))]

        self.window = None

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.reward_decay = reward_decay
        self.seed(seed=seed)
        self.agent_spawn_kwargs = agent_spawn_kwargs
        self.ghost_mode = ghost_mode

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

        return self.np_random.randint(0, 2) == 0

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def reset(self, **kwargs):
        for agent in self.agents:
            agent.agents = []
            agent.reset(new_episode=True)

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
        # agent_pos = agent.get_view_pos()
        grid = self.grid.slice(
            topX, topY, agent.view_size, agent.view_size, rot_k=agent.dir + 1
        )

        # Process occluders and visibility
        # Note that this incurs some performance cost
        # print(grid.opacity)
        vis_mask = agent.process_vis(grid.opacity)
        # print(vis_mask)

        # Warning about the rest of the function:
        #  Allows masking away objects that the agent isn't supposed to see.
        #  But breaks consistency between the states of the grid objects in the parial views
        #   and the grid objects overall.
        if len(getattr(agent, 'hide_item_types', []))>0:
            for i in range(grid.width):
                for j in range(grid.height):
                    item = grid.get(i,j)
                    if (item is not None) and (item is not agent) and (item.type in agent.hide_item_types):
                        if len(item.agents) > 0:
                            grid.set(i,j,item.agents[0])
                        else:
                            grid.set(i,j,None)

        return grid, vis_mask

    def gen_agent_obs(self, agent):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        grid, vis_mask = self.gen_obs_grid(agent)
        grid_image = grid.render(tile_size=agent.view_tile_size, visible_mask=vis_mask, top_agent=agent)
        if agent.observation_style=='image':
            return grid_image
        else:
            ret = {'pov': grid_image}
            if agent.observe_rewards:
                ret['reward'] = getattr(agent, 'step_reward', 0)
            if agent.observe_position:
                ret['position'] = np.array(agent.pos)/np.array([self.width, self.height], dtype=np.float)
            if agent.observe_orientation:
                ret['orientation'] = agent.dir
            return ret

    def gen_obs(self):
        return [self.gen_agent_obs(agent) for agent in self.agents]

    def __str__(self):
        return self.grid.__str__()

    def step(self, actions):

        def test_integrity(title=''):
            '''
            This function checks whether each agent is present in the grid in exactly one place.
            This is particularly helpful for validating the world state when ghost_mode=False and
            agents can stack.
            '''
            agent_locs = [[] for _ in range(len(self.agents))]
            for i in range(self.grid.width):
                for j in range(self.grid.height):
                    x = self.grid.get(i,j)
                    for k,agent in enumerate(self.agents):
                        if x==agent:
                            agent_locs[k].append(('top', (i,j)))
                        if hasattr(x, 'agents') and agent in x.agents:
                            agent_locs[k].append(('stacked', (i,j)))
            if not all([len(x)==1 for x in agent_locs]):
                print(f"{title} > Failed integrity test!")
                for a, al in zip(self.agents, agent_locs):
                    print(" > ", a.color,'-', al)
                import pdb; pdb.set_trace()

        assert len(actions) == len(self.agents)

        step_rewards = np.zeros((len(self.agents,)), dtype=np.float)

        self.step_count += 1

        iter_agents = list(enumerate(zip(self.agents, actions)))
        iter_order = np.arange(len(iter_agents))
        self.np_random.shuffle(iter_order)
        for shuffled_ix in iter_order:
            agent_no, (agent, action) = iter_agents[shuffled_ix]
            agent.step_reward = 0

            if agent.active:

                cur_pos = agent.pos[:]
                cur_cell = self.grid.get(*cur_pos)
                fwd_pos = agent.front_pos[:]
                fwd_cell = self.grid.get(*fwd_pos)
                agent_moved = False

                # Rotate left
                if action == agent.actions.left:
                    agent.dir = (agent.dir - 1) % 4

                # Rotate right
                elif action == agent.actions.right:
                    agent.dir = (agent.dir + 1) % 4

                # Move forward
                elif action == agent.actions.forward:
                    # Under the follow conditions, the agent can move forward.
                    can_move = fwd_cell is None or fwd_cell.can_overlap()
                    if self.ghost_mode is False and isinstance(fwd_cell, GridAgent):
                        can_move = False

                    if can_move:
                        agent_moved = True
                        # Add agent to new cell
                        if fwd_cell is None:
                            self.grid.set(*fwd_pos, agent)
                            agent.pos = fwd_pos
                        else:
                            fwd_cell.agents.append(agent)
                            agent.pos = fwd_pos

                        # Remove agent from old cell
                        if cur_cell == agent:
                            self.grid.set(*cur_pos, None)
                        else:
                            assert cur_cell.can_overlap()
                            cur_cell.agents.remove(agent)

                        # Add agent's agents to old cell
                        for left_behind in agent.agents:
                            cur_obj = self.grid.get(*cur_pos)
                            if cur_obj is None:
                                self.grid.set(*cur_pos, left_behind)
                            elif cur_obj.can_overlap():
                                cur_obj.agents.append(left_behind)
                            else: # How was "agent" there in teh first place?
                                raise ValueError("?!?!?!")

                        # After moving, the agent shouldn't contain any other agents.
                        agent.agents = [] 
                        # test_integrity(f"After moving {agent.color} fellow")

                        # Rewards can be got iff. fwd_cell has a "get_reward" method
                        if hasattr(fwd_cell, 'get_reward'):
                            rwd = fwd_cell.get_reward(agent)
                            if bool(self.reward_decay):
                                rwd *= (1.0-0.9*(self.step_count/self.max_steps))
                            step_rewards[agent_no] += rwd
                            agent.reward(rwd)
                            

                        if isinstance(fwd_cell, (Lava, Goal)):
                            agent.done = True

                # TODO: verify pickup/drop/toggle logic in an environment that 
                #  supports the relevant interactions.
                # Pick up an object
                elif action == agent.actions.pickup:
                    if fwd_cell and fwd_cell.can_pickup():
                        if agent.carrying is None:
                            agent.carrying = fwd_cell
                            agent.carrying.cur_pos = np.array([-1, -1])
                            self.grid.set(*fwd_pos, None)
                    else:
                        pass

                # Drop an object
                elif action == agent.actions.drop:
                    if not fwd_cell and agent.carrying:
                        self.grid.set(*fwd_pos, agent.carrying)
                        agent.carrying.cur_pos = fwd_pos
                        agent.carrying = None
                    else:
                        pass

                # Toggle/activate an object
                elif action == agent.actions.toggle:
                    if fwd_cell:
                        wasted = bool(fwd_cell.toggle(agent, fwd_pos))
                    else:
                        pass

                # Done action (not used by default)
                elif action == agent.actions.done:
                    pass

                else:
                    raise ValueError(f"Environment can't handle action {action}.")

                agent.on_step(fwd_cell if agent_moved else None)

        if self.step_count >= self.max_steps:
            dones = [True for agent in self.agents]
        else:
            dones = []
            for agent in self.agents:
                if agent.done and self.respawn:
                    resting_place_obj = self.grid.get(*agent.pos)
                    if resting_place_obj == agent:
                        if agent.agents:
                            self.grid.set(*agent.pos, agent.agents[0])
                            agent.agents[0].agents += agent.agents[1:]
                        else:
                            self.grid.set(*agent.pos, None)
                    else:
                        resting_place_obj.agents.remove(agent)
                        resting_place_obj.agents += agent.agents[:]
                        agent.agents = []
                            
                    agent.reset()
                    self.place_agent(agent, **self.agent_spawn_kwargs)
                dones.append(agent.done)

        done = np.array(dones, dtype=np.bool)
        if self.done_condition == "any":
            done = done.any()
        elif self.done_condition == "all":
            done = done.all()

        obs = [self.gen_agent_obs(agent) for agent in self.agents]

        return obs, step_rewards, done, {}

    @property
    def agent_positions(self):
        return [
            tuple(agent.pos) if agent.pos is not None else None for agent in self.agents
        ]

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        max_tries = int(max(1, min(max_tries, 1e5)))
        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))
        if size is None:
            size = (self.grid.width, self.grid.height)

        agent_positions = self.agent_positions
        for try_no in range(max_tries):
            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            if (
                (self.grid.get(*pos) is None)
                and (pos not in agent_positions)
                and (reject_fn is None or (not reject_fn(pos)))
            ):
                break
        else:
            raise RecursionError("Rejection sampling failed in place_obj.")

        self.grid.set(*pos, obj)
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

    def place_agent(self, agent, top=None, size=None, rand_dir=True, max_tries=1000):
        agent.pos = self.place_obj(agent, top=top, size=size, max_tries=max_tries)
        if rand_dir:
            agent.dir = self._rand_int(0, 4)
        return agent

    def place_agents(self, top=None, size=None, rand_dir=True, max_tries=1000):
        for agent in self.agents:
            self.place_agent(
                agent, top=top, size=size, rand_dir=rand_dir, max_tries=max_tries
            )
            if hasattr(self, "mission"):
                agent.mission = self.mission

    def render(
        self,
        mode="human",
        close=False,
        highlight=True,
        tile_size=TILE_PIXELS,
        show_agent_views=True,
        max_agents_per_col=3,
    ):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == "human" and not self.window:
            from gym.envs.classic_control.rendering import SimpleImageViewer

            self.window = SimpleImageViewer()

        # Compute which cells are visible to the agent
        highlight_mask = np.full((self.width, self.height), False, dtype=np.bool)
        for agent in self.agents:
            xlow, ylow, xhigh, yhigh = agent.get_view_exts()
            if agent.active:
                dxlow, dylow = max(0, 0-xlow), max(0, 0-ylow)
                dxhigh, dyhigh = max(0, xhigh-self.grid.width), max(0, yhigh-self.grid.height)
                if agent.see_through_walls:
                    highlight_mask[xlow+dxlow:xhigh-dxhigh, ylow+dylow:yhigh-dyhigh] = True
                else:
                    a,b = self.gen_obs_grid(agent)
                    highlight_mask[xlow+dxlow:xhigh-dxhigh, ylow+dylow:yhigh-dyhigh] |= (
                        rotate_grid(b, a.orientation)[dxlow:(xhigh-xlow)-dxhigh, dylow:(yhigh-ylow)-dyhigh]
                    )


        # Render the whole grid
        img = self.grid.render(
            tile_size, highlight_mask=highlight_mask if highlight else None
        )
        rescale = lambda X, rescale_factor=2: np.kron(
            X, np.ones((rescale_factor, rescale_factor, 1))
        )

        if show_agent_views:
            agent_no = 0
            cols = []
            rescale_factor = None
            pad = 5
            pad_grey = 100
            # img = np.pad(img, ((pad,pad),(pad,pad),(0,0)), constant_values=128)

            for col_no in range(len(self.agents) // (max_agents_per_col + 1) + 1):
                col_count = min(max_agents_per_col, len(self.agents) - agent_no)
                views = []

                for row_no in range(col_count):
                    tmp = self.gen_agent_obs(self.agents[agent_no])
                    if isinstance(tmp, dict) and 'pov' in tmp:
                        tmp = tmp['pov']
                    if rescale_factor is None:
                        rescale_factor = max(1, int(min(
                            (img.shape[0]/min(3, col_count)-pad) / (tmp.shape[1]),
                            (img.shape[1]-2*pad) // (2*(tmp.shape[1]))
                        ))-1)
                    views.append(
                        np.pad(
                            rescale(tmp, rescale_factor),
                            ((pad,pad),(pad,pad),(0,0)), constant_values=pad_grey))
                    agent_no += 1
                
                col_width = min(img.shape[1]//2, max([v.shape[1] for v in views]))+pad
                img_col = np.zeros((img.shape[0], col_width, 3), dtype=np.uint8)+pad_grey
                for k, view in enumerate(views):
                    start_x = (k * img.shape[0]) // len(views)
                    start_y = 0  # (k*img.shape[1])//len(views)
                    dx, dy = view.shape[:2]

                    tmp = img_col[start_x : start_x + dx, start_y : start_y + dy, :].shape
                    img_col[start_x : start_x + dx, start_y : start_y + dy, :] = view[:tmp[0],:tmp[1],:]
                cols.append(img_col)
            img = np.concatenate((img, *cols), axis=1)

        if mode == "human":
            self.window.imshow(img)

        return img
