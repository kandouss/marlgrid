import pyglet
from pyglet.gl import *
import sys

class SimpleImageViewer(object):
    def __init__(self, display=None, caption=None, maxwidth=500):
        self.window = None
        self.isopen = False
        self.display = display
        self.maxwidth = maxwidth
        self.caption = caption

    def imshow(self, arr):
        if self.window is None:
            height, width, _channels = arr.shape
            if width > self.maxwidth:
                scale = self.maxwidth / width
                width = int(scale * width)
                height = int(scale * height)
            self.window = pyglet.window.Window(width=width, height=height,
                display=self.display, vsync=False, resizable=True, caption=self.caption)
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"

        image = pyglet.image.ImageData(arr.shape[1], arr.shape[0],
            'RGB', arr.tobytes(), pitch=arr.shape[1]*-3)
        gl.glTexParameteri(gl.GL_TEXTURE_2D,
            gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture = image.get_texture()

        aspect_ratio = arr.shape[1]/arr.shape[0]
        forced_width = min(self.width, self.height * aspect_ratio)
        texture.height = int(forced_width / aspect_ratio)
        texture.width = int(forced_width)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        texture.blit(0, 0) # draw
        self.window.flip()
        
    def close(self):
        if self.isopen and sys.meta_path:
            # ^^^ check sys.meta_path to avoid 'ImportError: sys.meta_path is None, Python is likely shutting down'
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


class InteractivePlayerWindow(SimpleImageViewer):
    def __init__(self, display=None, caption=None, maxwidth=500):
        super().__init__(display=display, caption=caption, maxwidth=maxwidth)
        self.key = None
        self.action_count = 0

        self.action_map = {
            pyglet.window.key._0:0,
            pyglet.window.key._1:1,
            pyglet.window.key._2:2,
            pyglet.window.key._3:3,
            pyglet.window.key._4:4,
            pyglet.window.key._5:5,
            pyglet.window.key._6:6,
            pyglet.window.key.LEFT:0,
            pyglet.window.key.RIGHT:1,
            pyglet.window.key.UP:2,
            # pyglet.window.key.Q:-1,
        }

    def get_action(self, obs):
        if self.window is None:
            self.imshow(obs)

            @self.window.event
            def on_key_press(symbol, modifiers):
                self.key = symbol

            return self.get_action(obs)
    
        self.imshow(obs)
        self.key = None
        while self.key not in self.action_map:
            self.window.dispatch_events()
            pyglet.clock.tick()

        return self.action_map[self.key]