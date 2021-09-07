from typing import Optional, Tuple

import wx

from morphzero.ui.common import GameGraphicsContext


class Painter:
    """Painter knows how to paint (fill, strike, etc) objects in 1x1 area.

    To actually paint, we need to call transform_and_paint method, which transforms all paths
    (scales, translates, etc), paints, and reverts transformation (for future usage of the paths).

    Attributes:
        paths: Paths to be painted.
    """

    paths: Tuple[wx.GraphicsPath, ...]

    def __init__(self, *paths: wx.GraphicsPath):
        self.paths = paths

    def transform_and_paint(self, gc: wx.GraphicsContext, *transformations: wx.GraphicsMatrix) -> None:
        for path in self.paths:
            for transformation in transformations:
                path.Transform(transformation)
        self.paint(gc)
        for path in self.paths:
            for transformation in reversed(transformations):
                transformation.Invert()
                path.Transform(transformation)
                transformation.Invert()

    def paint(self, gc: wx.GraphicsContext) -> None:
        """Paints using GraphicsContext."""
        raise NotImplementedError()


class SimplePainter(Painter):
    """Fills (if brush is not None) and strokes (if pen is not None) each path.

    Attributes:
        brush: Brush to fill with (can be None).
        pen: Pen to stroke with (can be None).
    """
    brush: Optional[wx.Brush]
    pen: Optional[wx.Pen]

    def __init__(self,
                 brush: Optional[wx.Brush],
                 pen: Optional[wx.Pen],
                 *paths: wx.GraphicsPath):
        super().__init__(*paths)
        self.brush = brush
        self.pen = pen

    def paint(self, gc: wx.GraphicsContext) -> None:
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
    """Paints full circle without border."""

    def __init__(self, game_graphics_context: GameGraphicsContext, color: wx.Colour):
        path = game_graphics_context.graphics_renderer.CreatePath()
        path.AddEllipse(0.1, 0.1, 0.8, 0.8)
        super().__init__(wx.TheBrushList.FindOrCreateBrush(color),
                         None,  # pen,
                         path)
