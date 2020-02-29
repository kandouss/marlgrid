import gym
import numpy as np
import os

def export_video(X, outfile, fps=30, rescale_factor=2, repeat_last=0, progress_height=7):

    if isinstance(X, list):
        X = np.stack(X)

    if isinstance(X, np.float) and X.max() < 1:
        X = (X*255).astype(np.uint8).clip(0,255)
    
    if rescale_factor is not None and rescale_factor != 1:
        X = np.kron(X, np.ones((1, rescale_factor, rescale_factor, 1)))

    def make_frame(i):
        out = X[i]
        return out

    getframe = lambda t: make_frame(min(int(t*fps), len(X)-1))
    clip = mpy.VideoClip(getframe, duration=(len(X)-1+repeat_last)/fps)

    outfile = os.path.abspath(os.path.expanduser(outfile))
    if not os.path.isdir(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    clip.write_videofile(outfile, fps=fps)

class GridRecorder(gym.core.Wrapper):
    def __init__(self, env, max_len=1000):
        super().__init__(env)


        try:
            import moviepy.editor as mpy
        except:
            raise ImportError("GridRecorder requires moviepy library. Try installing:\n $ pip install moviepy")

        self.frames = None
        self.ptr = 0
        self.recording = False
        if hasattr(env, 'max_steps') and env.max_steps != 0:
            self.max_len = env.max_steps
        else:
            self.max_len = max_len
    
    def reset(self, **kwargs):
        self.ptr = 0
        return self.env.reset(**kwargs)

    def append_current_frame(self):
        if self.recording:
            new_frame = self.env.render(mode='rgb_array')
            if self.frames is None:
                self.frames = np.zeros((self.max_len, *new_frame.shape), dtype=new_frame.dtype)
            self.frames[self.ptr] = new_frame
            self.ptr += 1


    def step(self, action):
        self.append_current_frame()
        obs, rew, done, info = self.env.step(action)
        # if done:
        #     self.append_current_frame()
        return obs, rew, done, info

    def export_video(self, output_filename, fps=20, rescale_factor=1, repeat_last=0, **kwargs):
        if self.recording:
            return export_video(self.frames[:self.ptr], output_filename, fps=fps, rescale_factor=rescale_factor, repeat_last=repeat_last, **kwargs)
