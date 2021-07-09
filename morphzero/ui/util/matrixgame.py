from __future__ import annotations

import wx

from morphzero.core.common.connect_on_matrix_board import ConnectOnMatrixBoardState, ConnectOnMatrixBoardRules
from morphzero.core.common.matrix_board import MatrixBoardSize, MatrixBoardCoordinates
from morphzero.core.game import Player
from morphzero.core.game_service import GameService
from morphzero.ui.common import GameGraphicsContext
from morphzero.ui.util.painter import SimplePainter, Painter


def recommended_game_board_size(board_size: MatrixBoardSize) -> wx.Size:
    cell_size = 100 if max(board_size) < 7 else 50
    return wx.Size(board_size[1] * cell_size, board_size[0] * cell_size)


def get_cell_size(board_size: MatrixBoardSize, window_size: wx.Size) -> tuple[float, float]:
    rows, columns = board_size
    width, height = window_size
    return width / columns, height / rows


class MatrixGameBoard(wx.Panel):
    """The Base class for Panel that draws board for Matrix games."""
    game_service: GameService
    on_click_callback: OnClickCallback
    grid_painter: Painter
    painters: dict[Player, Painter]
    mouse_event_manager: MatrixGameBoard.MouseEventsManager

    def __init__(self,
                 parent: wx.Window,
                 game_service: GameService,
                 on_click_callback: OnClickCallback,
                 grid_painter: Painter,
                 painters: dict[Player, Painter]):
        super().__init__(parent)
        self.game_service = game_service
        self.on_click_callback = on_click_callback
        self.grid_painter = grid_painter
        self.painters = painters

        # events
        self.mouse_event_manager = MatrixGameBoard.MouseEventsManager(self)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)

    def DoGetBestSize(self) -> wx.Size:
        return recommended_game_board_size(self.get_board_size())

    def get_rules(self) -> ConnectOnMatrixBoardRules:
        rules = self.game_service.engine.rules
        assert isinstance(rules, ConnectOnMatrixBoardRules)
        return rules

    def get_board_size(self) -> MatrixBoardSize:
        return self.get_rules().board_size

    def get_cell_size(self) -> wx.Size:
        return wx.Size(*get_cell_size(self.get_board_size(), self.GetClientSize()))

    def mouse_position_to_board_coordinates(self, position: wx.Point) -> MatrixBoardCoordinates:
        cell_size = self.get_cell_size()
        x, y = position.Get()
        return MatrixBoardCoordinates(row=int(y // cell_size.GetHeight()),
                                      column=int(x // cell_size.GetWidth()))

    # Drawing
    def paint(self, gc: wx.GraphicsContext) -> None:
        """Paints entire panel."""
        self.draw_grid(gc)
        self.draw_pieces(gc)

    def draw_grid(self, gc: wx.GraphicsContext) -> None:
        """Draws the grid."""
        transformation = gc.CreateMatrix()
        transformation.Scale(*self.GetClientSize())
        self.grid_painter.transform_and_paint(gc, transformation)

    def draw_pieces(self, gc: wx.GraphicsContext) -> None:
        """Draws pieces."""
        state = self.game_service.state
        assert isinstance(state, ConnectOnMatrixBoardState)

        board_size = self.get_board_size()
        cell_size = self.get_cell_size()
        for row in range(board_size.rows):
            for column in range(board_size.columns):
                painter = self.painters.get(state.board[row, column])
                if painter:
                    transformation = gc.CreateMatrix()
                    transformation.Scale(cell_size.GetWidth(), cell_size.GetHeight())
                    transformation.Translate(column, row)
                    painter.transform_and_paint(gc, transformation)

    # Events
    def on_paint(self, event: wx.PaintEvent) -> None:
        gc = wx.GraphicsContext.Create(wx.PaintDC(self))
        self.paint(gc)
        event.Skip()

    def on_click(self, board_coordinates: MatrixBoardCoordinates) -> None:
        wx.CallAfter(self.on_click_callback.on_click, board_coordinates)

    def on_destroy(self, _: wx.WindowDestroyEvent) -> None:
        self.mouse_event_manager.Destroy()

    class MouseEventsManager(wx.MouseEventsManager):
        board: MatrixGameBoard

        def __init__(self, board: MatrixGameBoard):
            super().__init__(board)
            self.board = board

        def board_coordinates_to_item(self, board_coordinates: MatrixBoardCoordinates) -> int:
            board_size = self.board.get_board_size()
            return board_coordinates.row * board_size.columns + board_coordinates.column

        def item_to_board_coordinates(self, item: int) -> MatrixBoardCoordinates:
            _, columns = self.board.get_board_size()
            return MatrixBoardCoordinates(row=item // columns,
                                          column=item % columns)

        def MouseClicked(self, item: int) -> bool:
            board_coordinates = self.item_to_board_coordinates(item)
            self.board.on_click(board_coordinates)
            return True

        def MouseHitTest(self, position: wx.Point) -> int:
            board_coordinates = self.board.mouse_position_to_board_coordinates(position)
            item = self.board_coordinates_to_item(board_coordinates)
            return item

        def MouseDragBegin(self, _item: int, _position: wx.Point) -> bool:
            return False

    class AdditionalDrawing:
        def draw(self, gc: wx.GraphicsContext) -> None:
            raise NotImplementedError()

    class OnClickCallback:
        def on_click(self, board_coordinates: MatrixBoardCoordinates) -> None:
            raise NotImplementedError()


class GridPainter(SimplePainter):
    def __init__(self, game_graphics_context: GameGraphicsContext, line_width: int = 2):
        rules = game_graphics_context.game_config.rules
        assert isinstance(rules, ConnectOnMatrixBoardRules)

        board_size = rules.board_size
        cell_width, cell_height = get_cell_size(board_size, wx.Size(1, 1))

        path = game_graphics_context.graphics_renderer.CreatePath()
        for column in range(1, board_size.columns):
            path.MoveToPoint(column * cell_width, 0)
            path.AddLineToPoint(column * cell_width, 1)
        for row in range(1, board_size.rows):
            path.MoveToPoint(0, row * cell_height)
            path.AddLineToPoint(1, row * cell_height)
        super().__init__(None,  # brush
                         wx.ThePenList.FindOrCreatePen(wx.BLACK, line_width),
                         path)
