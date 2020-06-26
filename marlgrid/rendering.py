import pyglet
from pyglet.gl import *

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
        texture.width = self.width
        texture.height = self.height
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
