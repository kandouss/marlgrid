
import gym
import gym_minigrid
import numpy as np
import moviepy.editor as mpy

def make_progress_bar(i, n, w, h, rgb=(50,90,50), wpad=2, hpad=1):
    x = np.zeros((w,h,3))
    x[wpad:-wpad,hpad:-hpad][
        np.arange(w-2*wpad)/(w-2*wpad) >= (i/n), :, :
    ] = rgb
    return x

def export_video(X, outfile, fps=30, rescale_factor=2, repeat_last=0, progress_bar=True, progress_height=7):
    if isinstance(X, list):
        X = np.stack(X)

    if isinstance(X, np.float) and X.max() < 1:
        X = (X*255).astype(np.uint8).clip(0,255)
    
    X = np.kron(X, np.ones((1, rescale_factor, rescale_factor, 1)))
    def make_frame(i):
        out = X[i]
        if progress_bar:
            out[-progress_height:,:] = np.swapaxes(make_progress_bar(i, X.shape[0], X.shape[1], progress_height),0,1)
        return out

    getframe = lambda t: make_frame(min(int(t*fps), len(X)-1))
    clip = mpy.VideoClip(getframe, duration=(len(X)-1+repeat_last)/fps)
    clip.write_videofile(outfile, fps=fps)

class GridRecorder(gym.core.Wrapper):
    def __init__(self, env, tile_size=11):
        if not isinstance(env, gym_minigrid.minigrid.MiniGridEnv):
            raise ValueError(f"{self.__class__.__name__} only supports minigrid environments.")
        super().__init__(env)
        self.tile_size = tile_size
        self.frames = []

    def reset(self, **kwargs):
        self.frames = []
        return self.env.reset(**kwargs)

    def append_current_frame(self):
        self.frames.append(
            self.env.render(tile_size=self.tile_size, mode='not_human'))

    def step(self, action):
        self.append_current_frame()
        obs, rew, done, info = self.env.step(action)
        if done:
            self.append_current_frame()
        return obs, rew, done, info

    def export_video(self, output_filename, fps=20, rescale_factor=2, repeat_last=40, **kwargs):
        return export_video(self.frames, output_filename, fps=fps, rescale_factor=rescale_factor, repeat_last=repeat_last, **kwargs)


env = gym_minigrid.envs.empty.EmptyEnv(size=10)
env.max_steps = 200
env = GridRecorder(env, tile_size=11)

obs = env.reset()

count=0
done = False
while not done:
    act = env.action_space.sample()
    obs, rew, done, _ = env.step(act)
    count += 1

print(f"Done after {count} frames!")
env.export_video('test.mp4', progress_bar=True, progress_height=7)