import gym
import time

from abc import ABCMeta, abstractmethod, ABC
from contextlib import contextmanager, ExitStack

from .env.base import GridAgent

class LearningAgent(GridAgent):#, ABC):
    def __init__(self, *args, **kwargs):
        # GridAgent init notably sets self.action_space, self.observation_space
        super().__init__(*args, **kwargs)
        self.last_obs = None

        self.n_episodes = 0

    def action_step(self, obs):
        raise NotImplementedError

    def save_step(self, *values):
        raise NotImplementedError

    def start_episode(self):
        raise NotImplementedError

    def end_episode(self):
        raise NotImplementedError

    @contextmanager
    def episode(self):
        self.start_episode()
        yield self
        self.end_episode()
        self.n_episodes += 1


class IndependentLearners(LearningAgent):
    def __init__(self, *agents):
        self.agents = list(agents)
        self.observation_space = self.combine_spaces([agent.observation_space for agent in agents])
        self.action_space = self.combine_spaces([agent.action_space for agent in agents])

    def combine_spaces(self, spaces):
        # if all(isinstance(space, gym.spaces.Discrete) for space in spaces):
        #     return gym.spaces.MultiDiscrete([space.n for space in spaces])
        return gym.spaces.Tuple(tuple(spaces))

    def action_step(self, obs_array):
        return [agent.action_step(obs) if agent.active else agent.action_space.sample()
                for agent, obs in zip(self.agents, obs_array)]

    def save_step(self, *values):
        values = [
            v if hasattr(v, '__len__') and len(v) == len(self.agents) 
            else [v for _ in self.agents] 
            for v in values]

        for agent, agent_values in zip(self.agents, zip(*values)):
            agent.save_step(*agent_values)

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    @property
    def episode_duration(self):
        if self.start_time is None:
            return 0
        return (time.time() - self.start_time)

    @property
    def active(self):
        return np.array([agent.active for agent in self.agents], dtype=np.bool)

    @contextmanager    
    def episode(self):
        self.start_time = time.time()
        with ExitStack() as stack:
            for agent in self.agents:
                try:
                    stack.enter_context(agent.episode())
                except:
                    import pdb; pdb.set_trace()
            yield self

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, key):
        return self.agents[key]

    def __iter__(self):
        return self.agents.__iter__()