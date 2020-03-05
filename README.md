# marlgrid
Gridworld for MARL experiments

## Training multiple independent learners

Create an RL agent (subclassing `marlgrid.agents.LearningAgent`) that implements:
 - `action_step(self, obs)`,
 - `save_step(self, *transition_values)`,
 - `start_episode(self)` (optional),
 - `end_episode(self)` (optional),

 e.g. `TestRLAgent`.
 
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