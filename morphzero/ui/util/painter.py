import wx


class Painter:
    """
    Painter knows how to paint (fill, strike, etc) objects in 1x1 area.
    To actually paint, we need to call transform_and_paint method, which transforms all paths
    (scales, translates, etc), paints, and reverts transformation (for future usage).
    """

    def __init__(self, *paths):
        self.paths = paths

    def transform_and_paint(self, gc, *transformations):
        for path in self.paths:
            for transformation in transformations:
                path.Transform(transformation)
        self.paint(gc)
        for path in self.paths:
            for transformation in reversed(transformations):
                transformation.Invert()
                path.Transform(transformation)
                transformation.Invert()

    def paint(self, gc):
        raise NotImplementedError()


class SimplePainter(Painter):
    """
    Fills (if brush is not None) and strokes (if pen is not None) each path.
    """

    def __init__(self, brush, pen, *args):
        super().__init__(*args)
        self.brush = brush
        self.pen = pen

    def paint(self, gc):
        if self.brush:
            gc.SetBrush(self.brush)
        if self.pen:
            gc.SetPen(self.pen)
        for path in self.paths:
            if self.brush:
                gc.FillPath(path)
            if self.pen:
                gc.StrokePath(path)


class CirclePainter(SimplePainter):
    def __init__(self, game_graphics_context, color):
        path = game_graphics_context.graphics_renderer.CreatePath()
        path.AddEllipse(0.1, 0.1, 0.8, 0.8)
        super().__init__(wx.TheBrushList.FindOrCreateBrush(color),
                         None,  # pen,
                         path)
