import numpy as np
from enum import IntEnum
from gym_minigrid.rendering import (
    fill_coords,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

# Map of color names to RGB values
COLORS = {
    "red": np.array([255, 0, 0]),
    "orange": np.array([255, 165, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "cyan": np.array([0, 139, 139]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "olive": np.array([128, 128, 0]),
    "grey": np.array([100, 100, 100]),
    "worst": np.array([74, 65, 42]),  # https://en.wikipedia.org/wiki/Pantone_448_C
    "pink": np.array([255, 0, 189]),
    "white": np.array([255,255,255]),
    "prestige": np.array([255,255,255]),
    "shadow": np.array([35,25,30]), # nice dark purpley color for cells agents can't see.
}

# Used to map colors to integers
COLOR_TO_IDX = dict({v: k for k, v in enumerate(COLORS.keys())})

OBJECT_TYPES = []

class RegisteredObjectType(type):
    def __new__(meta, name, bases, class_dict):
        cls = type.__new__(meta, name, bases, class_dict)
        if name not in OBJECT_TYPES:
            OBJECT_TYPES.append(cls)

        def get_recursive_subclasses():
            return OBJECT_TYPES

        cls.recursive_subclasses = staticmethod(get_recursive_subclasses)
        return cls


class WorldObj(metaclass=RegisteredObjectType):
    def __init__(self, color="worst", state=0):
        self.color = color
        self.state = state
        self.contains = None

        self.agents = [] # Some objects can have agents on top (e.g. floor, open doors, etc).
        
        self.pos_init = None
        self.pos = None
        self.is_agent = False

    @property
    def dir(self):
        return None

    def set_position(self, pos):
        if self.pos_init is None:
            self.pos_init = pos
        self.pos = pos

    @property
    def numeric_color(self):
        return COLORS[self.color]
    
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

    def encode(self, str_class=False):
        # Note 5/29/20: Commented out the condition below; was causing agents to 
        #  render incorrectly in partial views. In particular, if there were N red agents,
        #  agents {i != k} would render as blue (rather than red) in agent k's partial view.
        # # if len(self.agents)>0:
        # #     return self.agents[0].encode(str_class=str_class)
        # # else:
        enc_class = self.type if bool(str_class) else self.recursive_subclasses().index(self.__class__)
        enc_color = self.color if isinstance(self.color, int) else COLOR_TO_IDX[self.color]
        return (enc_class, enc_color, self.state)

    def describe(self):
        return f"Obj: {self.type}({self.color}, {self.state})"

    @classmethod
    def decode(cls, type, color, state):
        if isinstance(type, str):
            cls_subclasses = {c.__name__: c for c in cls.recursive_subclasses()}
            if type not in cls_subclasses:
                raise ValueError(
                    f"Not sure how to construct a {cls} of (sub)type {type}"
                )
            return cls_subclasses[type](color, state)
        elif isinstance(type, int):
            subclass = cls.recursive_subclasses()[type]
            return subclass(color, state)

    def render(self, img):
        raise NotImplementedError

    def str_render(self, dir=0):
        return "??"


class GridAgent(WorldObj):
    def __init__(self, *args, color='red', **kwargs):
        super().__init__(*args, **{'color':color, **kwargs})
        self.metadata = {
            'color': color,
        }
        self.is_agent = True

    @property
    def dir(self):
        return self.state % 4

    @property
    def type(self):
        return 'Agent'

    @dir.setter
    def dir(self, dir):
        self.state = self.state // 4 + dir % 4

    def str_render(self, dir=0):
        return [">>", "VV", "<<", "^^"][(self.dir + dir) % 4]

    def can_overlap(self):
        return True

    def render(self, img):
        tri_fn = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81),)
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * np.pi * (self.dir))
        fill_coords(img, tri_fn, COLORS[self.color])


class BulkObj(WorldObj, metaclass=RegisteredObjectType):
    # Todo: special behavior for hash, eq if the object has an agent.
    def __hash__(self):
        return hash((self.__class__, self.color, self.state, tuple(self.agents)))

    def __eq__(self, other):
        return hash(self) == hash(other)

class BonusTile(WorldObj):
    def __init__(self, reward, penalty=-0.1, bonus_id=0, n_bonus=1, initial_reward=True, reset_on_mistake=False, color='yellow', *args, **kwargs):
        super().__init__(*args, **{'color': color, **kwargs, 'state': bonus_id})
        self.reward = reward
        self.penalty = penalty
        self.n_bonus = n_bonus
        self.bonus_id = bonus_id
        self.initial_reward = initial_reward
        self.reset_on_mistake = reset_on_mistake

    def can_overlap(self):
        return True

    def str_render(self, dir=0):
        return "BB"

    def get_reward(self, agent):
        # If the agent hasn't hit any bonus tiles, set its bonus state so that
        #  it'll get a reward from hitting this tile.
        first_bonus = False
        if agent.bonus_state is None:
            agent.bonus_state = (self.bonus_id - 1) % self.n_bonus
            first_bonus = True

        if agent.bonus_state == self.bonus_id:
            # This is the last bonus tile the agent hit
            rew = -np.abs(self.penalty)
        elif (agent.bonus_state + 1)%self.n_bonus == self.bonus_id:
            # The agent hit the previous bonus tile before this one
            agent.bonus_state = self.bonus_id
            # rew = agent.bonus_value
            rew = self.reward
        else:
            # The agent hit any other bonus tile before this one
            rew = -np.abs(self.penalty)

        if self.reset_on_mistake:
            agent.bonus_state = self.bonus_id

        if first_bonus and not bool(self.initial_reward):
            return 0
        else:
            return rew

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Goal(WorldObj):
    def __init__(self, reward, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward = reward

    def can_overlap(self):
        return True

    def get_reward(self, agent):
        return self.reward

    def str_render(self, dir=0):
        return "GG"

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    def can_overlap(self):
        return True# and self.agent is None

    def str_render(self, dir=0):
        return "FF"

    def render(self, img):
        # Give the floor a pale color
        c = COLORS[self.color]
        img.setLineColor(100, 100, 100, 0)
        img.setColor(*c / 2)
        # img.drawPolygon([
        #     (1          , TILE_PIXELS),
        #     (TILE_PIXELS, TILE_PIXELS),
        #     (TILE_PIXELS,           1),
        #     (1          ,           1)
        # ])


class EmptySpace(WorldObj):
    def can_verlap(self):
        return True

    def str_render(self, dir=0):
        return "  "


class Lava(WorldObj):
    def can_overlap(self):
        return True# and self.agent is None

    def str_render(self, dir=0):
        return "VV"

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(BulkObj):
    def see_behind(self):
        return False

    def str_render(self, dir=0):
        return "WW"

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Key(WorldObj):
    def can_pickup(self):
        return True

    def str_render(self, dir=0):
        return "KK"

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    def can_pickup(self):
        return True

    def str_render(self, dir=0):
        return "AA"

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Door(WorldObj):
    states = IntEnum("door_state", "open closed locked")

    def can_overlap(self):
        return self.state == self.states.open# and self.agent is None  # is open

    def see_behind(self):
        return self.state == self.states.open  # is open

    def toggle(self, agent, pos):
        if self.state == self.states.locked:  # is locked
            # If the agent is carrying a key of matching color
            if (
                agent.carrying is not None
                and isinstance(agent.carrying, Key)
                and agent.carrying.color == self.color
            ):
                self.state = self.states.closed
        elif self.state == self.states.closed:  # is unlocked but closed
            self.state = self.states.open
        elif self.state == self.states.open:  # is open
            self.state = self.states.closed
        return True

    def render(self, img):
        c = COLORS[self.color]

        if self.state == self.states.open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.state == self.states.locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

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
        return "BB"

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)
