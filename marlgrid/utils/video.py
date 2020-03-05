import gym
import numpy as np
import os
import tqdm


def export_video(X, outfile, fps=30, rescale_factor=2):

    try:
        import moviepy.editor as mpy
    except:
        raise ImportError(
            "GridRecorder requires moviepy library. Try installing:\n $ pip install moviepy"
        )

    if isinstance(X, list):
        X = np.stack(X)

    if isinstance(X, np.float) and X.max() < 1:
        X = (X * 255).astype(np.uint8).clip(0, 255)

    if rescale_factor is not None and rescale_factor != 1:
        X = np.kron(X, np.ones((1, rescale_factor, rescale_factor, 1)))

    def make_frame(i):
        out = X[i]
        return out

    getframe = lambda t: make_frame(min(int(t * fps), len(X) - 1))
    clip = mpy.VideoClip(getframe, duration=len(X) / fps)

    outfile = os.path.abspath(os.path.expanduser(outfile))
    if not os.path.isdir(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    clip.write_videofile(outfile, fps=fps)


def render_frames(X, path, ext="png"):

    try:
        from PIL import Image
    except ImportError as e:
        raise ImporErroR(
            "Error importing from PIL in export_frames. Try installing PIL:\n $ pip install Pillow"
        )

    # If the path has a file extension, dump frames in a new directory with = path minus extension
    if "." in os.path.basename(path):
        path = os.path.splitext(path)[0]
    if not os.path.isdir(path):
        os.makedirs(path)

    for k, frame in tqdm.tqdm(enumerate(X), total=len(X)):
        Image.fromarray(frame, "RGB").save(os.path.join(path, f"frame_{k}.{ext}"))


class GridRecorder(gym.core.Wrapper):
    default_max_len = 1000

    def __init__(self, env, max_len=None, render_kwargs={}):
        super().__init__(env)

        self.frames = None
        self.ptr = 0
        self.recording = False
        self.render_kwargs = render_kwargs
        if max_len is None:
            if hasattr(env, "max_steps") and env.max_steps != 0:
                self.max_len = env.max_steps + 1
            else:
                self.max_len = max_len + 1
        else:
            self.max_len = self.default_max_len + 1

    def reset(self, **kwargs):
        self.ptr = 0
        return self.env.reset(**kwargs)

    def append_current_frame(self):
        if self.recording:
            new_frame = self.env.render(mode="rgb_array", **self.render_kwargs)
            if self.frames is None:
                self.frames = np.zeros(
                    (self.max_len, *new_frame.shape), dtype=new_frame.dtype
                )
            self.frames[self.ptr] = new_frame
            self.ptr += 1

    def step(self, action):
        self.append_current_frame()
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info

    def export_video(
        self,
        output_filename,
        fps=20,
        rescale_factor=1,
        render_last=True,
        render_frame_images=False,
        **kwargs,
    ):
        if self.recording:
            if render_last:
                self.frames[self.ptr] = self.env.render(
                    mode="rgb_array", **self.render_kwargs
                )
            if render_frame_images:
                render_frames(self.frames[: self.ptr + 1], output_filename)
            return export_video(
                self.frames[: self.ptr + 1],
                output_filename,
                fps=fps,
                rescale_factor=rescale_factor,
                **kwargs,
            )
