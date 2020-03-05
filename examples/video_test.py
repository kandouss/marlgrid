from marlgrid.utils.video import GridRecorder
import gym_minigrid

env = gym_minigrid.envs.empty.EmptyEnv(size=10)
env.max_steps = 200

env = GridRecorder(env, render_kwargs={"tile_size": 11})

obs = env.reset()
env.recording = True

count = 0
done = False

while not done:
    act = env.action_space.sample()
    obs, rew, done, _ = env.step(act)
    count += 1

env.export_video("test_minigrid.mp4")
