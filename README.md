# MarlGrid
Gridworld for MARL experiments, based on [MiniGrid](https://github.com/maximecb/gym-minigrid).

[![Three agents navigating a cluttered MarlGrid environment.](https://img.youtube.com/vi/e0xL6KB6RBA/0.jpg)](https://youtube.com/watch?v=e0xL6KB6RBA)
<video src="https://kam.al/images/extra/cluttered_multigrid_example.mp4" id="spinning-video" controls preload loop style="width:400px; max-width:100%; display:block; margin-left:auto; margin-right:auto; margin-bottom:20px;"></video>

## Training multiple independent learners

### Pre-built environment

MarlGrid comes with a few pre-built environments (see marlgrid/envs):
- `MarlGrid-3AgentCluttered11x11-v0`
- `MarlGrid-3AgentCluttered15x15-v0`
- `MarlGrid-2AgentEmpty9x9-v0`
- `MarlGrid-3AgentEmpty9x9-v0`
- `MarlGrid-4AgentEmpty9x9-v0`
(as of v0.0.2)

### Custom environment

Create an RL agent (e.g. `TestRLAgent` subclassing `marlgrid.agents.LearningAgent`) that implements:
 - `action_step(self, obs)`,
 - `save_step(self, *transition_values)`,
 - `start_episode(self)` (optional),
 - `end_episode(self)` (optional),
 
Then multiple such agents can be trained in a MARLGrid environment like `ClutteredMultiGrid`:

```
agents = marlgrid.agents.IndependentLearners(
    TestRLAgent(),
    TestRLAgent(),
    TestRLAgent()
)

env = ClutteredMultiGrid(agents, grid_size=15, n_clutter=10)


for i_episode in range(N_episodes):

    obs_array = env.reset()

    with agents.episode():

        episode_over = False

        while not episode_over:
            # env.render()

            # Get an array with actions for each agent.
            action_array = agents.action_step(obs_array)

            # Step the multi-agent environment
            next_obs_array, reward_array, done, _ = env.step(action_array)

            # Save the transition data to replay buffers, if necessary
            agents.save_step(obs_array, action_array, next_obs_array, reward_array, done)

            obs_array = next_obs_array

            episode_over = done
            # or if "done" is per-agent:
            episode_over = all(done) # or any(done)
            
```