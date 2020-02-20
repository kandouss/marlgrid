
import gym

import gym_minigrid
import numpy as np



env = gym_minigrid.envs.empty.EmptyEnv(size=10)
env.max_steps = 200
# eng = GridRecorder(env, tile_size=11)

obs = env.reset()

count = 0
done = False
while not done:
    act = env.action_space.sample()
    obs, rew, done, _ = env.step(act)
    env.render()
    count += 1


print("All done.")