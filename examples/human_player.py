import numpy as np
import marlgrid

from marlgrid.rendering import InteractivePlayerWindow
from marlgrid.agents import GridAgentInterface
from marlgrid.envs import env_from_config

class HumanPlayer:
    def __init__(self):
        self.player_window = InteractivePlayerWindow(
            caption="interactive marlgrid"
        )
        self.episode_count = 0

    def action_step(self, obs):
        return self.player_window.get_action(obs.astype(np.uint8))

    def save_step(self, obs, act, rew, done):
        print(f"   step {self.step_count:<4d}: reward {rew} (episode total {self.cumulative_reward})")
        self.cumulative_reward += rew
        self.step_count += 1

    def start_episode(self):
        self.cumulative_reward = 0
        self.step_count = 0
    
    def end_episode(self):
        print(
            f"Finished episode {self.episode_count} after {self.step_count} steps."
            f"  Episode return was {self.cumulative_reward}."
        )
        self.episode_count += 1


env_config =  {
    "env_class": "ClutteredGoalCycleEnv",
    "grid_size": 13,
    "max_steps": 250,
    "clutter_density": 0.15,
    "respawn": True,
    "ghost_mode": True,
    "reward_decay": False,
    "n_bonus_tiles": 3,
    "initial_reward": True,
    "penalty": -1.5
}

player_interface_config = {
    "view_size": 7,
    "view_offset": 1,
    "view_tile_size": 11,
    "observation_style": "rich",
    "see_through_walls": False,
    "color": "prestige"
}

# Add the player/agent config to the environment config (as expected by "env_from_config" below)
env_config['agents'] = [player_interface_config]

# Create the environment based on the combined env/player config
env = env_from_config(env_config)

# Create a human player interface per the class defined above
human = HumanPlayer()

# Start an episode!
# Each observation from the environment contains a list of observaitons for each agent.
# In this case there's only one agent so the list will be of length one.
obs_list = env.reset()

human.start_episode()
done = False
while not done:

    env.render() # OPTIONAL: render the whole scene + birds eye view
    
    player_action = human.action_step(obs_list[0]['pov'])
    # The environment expects a list of actions, so put the player action into a list
    agent_actions = [player_action]

    next_obs_list, rew_list, done, _ = env.step(agent_actions)
    
    human.save_step(
        obs_list[0], player_action, rew_list[0], done
    )

    obs_list = next_obs_list

human.end_episode()
