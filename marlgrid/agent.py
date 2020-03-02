 import gym
import time

class MultiAgent:
    def __init__(self, *agents):
        self.agents = list(agents)
        self.observation_space = self.combine_spaces([agent.observation_space for agent in agents])
        self.action_space = self.combine_spaces([agent.action_space for agent in agents])

    def combine_spaces(self, spaces):
        # if all(isinstance(space, gym.spaces.Discrete) for space in spaces):
        #     return gym.spaces.MultiDiscrete([space.n for space in spaces])
        return gym.spaces.Tuple(tuple(spaces))

    def __len__(self):
        return len(self.agents)
    def __getitem__(self, key):
        return self.agents[key]
    def __iter__(self):
        return self.agents.__iter__()

    def action_from_obs(self, obs_array):
        return [agent.action_from_obs(obs) if agent.active else agent.action_space.sample()
                for agent, obs in zip(self.agents, obs_array)]

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
                agent.train()
                stack.enter_context(agent.episode())
            yield self