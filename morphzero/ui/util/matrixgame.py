import wx

from morphzero.common import BoardCoordinates
from morphzero.ui.util.painter import SimplePainter


def recommended_game_board_size(board_size):
    cell_size = 100 if max(board_size) < 7 else 50
    return board_size[1] * cell_size, board_size[0] * cell_size


def get_cell_size(board_size, window_size):
    rows, columns = board_size
    width, height = window_size
    return width / columns, height / rows


class MatrixGameBoard(wx.Panel):
    def __init__(self,
                 parent,
                 game_service,
                 on_click_callback,
                 grid_painter,
                 painters,
                 **kwargs):
        super().__init__(parent, **kwargs)
        self.game_service = game_service
        self.on_click_callback = on_click_callback
        self.grid_painter = grid_painter
        self.painters = painters

        # events
        self.mouse_event_manager = MatrixGameBoard.MouseEventsManager(self)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)

    def DoGetBestSize(self):
        return recommended_game_board_size(self.get_board_size())

    def get_board_size(self):
        return self.game_service.engine.rules.board_size

    def get_cell_size(self):
        return get_cell_size(self.get_board_size(), self.GetClientSize())

    def mouse_position_to_board_coordinates(self, position):
        cell_width, cell_height = self.get_cell_size()
        return BoardCoordinates(row=int(position.y // cell_height),
                                column=int(position.x // cell_width))

    # Drawing
    def paint(self, gc):
        self.draw_grid(gc)
        self.draw_pieces(gc)

    def draw_grid(self, gc):
        transformation = gc.CreateMatrix()
        transformation.Scale(*self.GetClientSize())
        self.grid_painter.transform_and_paint(gc, transformation)

    def draw_pieces(self, gc):
        state = self.game_service.state
        if state is None:
            return
        rows, columns = self.get_board_size()
        cell_width, cell_height = self.get_cell_size()
        for row in range(rows):
            for column in range(columns):
                painter = self.painters.get(state.board[row, column])
                if painter:
                    transformation = gc.CreateMatrix()
                    transformation.Scale(cell_width, cell_height)
                    transformation.Translate(column, row)
                    painter.transform_and_paint(gc, transformation)

    # Events
    def on_paint(self, event):
        gc = wx.GraphicsContext.Create(wx.PaintDC(self))
        self.paint(gc)
        event.Skip()

    def on_click(self, board_coordinates):
        wx.CallAfter(self.on_click_callback, board_coordinates)

    def on_destroy(self, _):
        self.mouse_event_manager.Destroy()

    class MouseEventsManager(wx.MouseEventsManager):
        def __init__(self, board, *args, **kw):
            super().__init__(board, *args, **kw)
            self.board = board

        def board_coordinates_to_item(self, board_coordinates):
            _, columns = self.board.get_board_size()
            return board_coordinates.row * columns + board_coordinates.column

        def item_to_board_coordinates(self, item):
            _, columns = self.board.get_board_size()
            return BoardCoordinates(row=item // columns,
                                    column=item % columns)

        def MouseClicked(self, item):
            board_coordinates = self.item_to_board_coordinates(item)
            self.board.on_click(board_coordinates)
            return True

        def MouseHitTest(self, position):
            board_coordinates = self.board.mouse_position_to_board_coordinates(position)
            item = self.board_coordinates_to_item(board_coordinates)
            return item

        def MouseDragBegin(self, _item, _position):
            return False

    class AdditionalDrawing:
        def draw(self, gc):
            raise NotImplementedError()


class GridPainter(SimplePainter):
    def __init__(self, game_graphics_context, line_width=2):
        rows, columns = game_graphics_context.game_config.rules.board_size
        cell_width, cell_height = get_cell_size((rows, columns), (1, 1))

        path = game_graphics_context.graphics_renderer.CreatePath()
        for column in range(1, columns):
            path.MoveToPoint(column * cell_width, 0)
            path.AddLineToPoint(column * cell_width, 1)
        for row in range(1, rows):
            path.MoveToPoint(0, row * cell_height)
            path.AddLineToPoint(1, row * cell_height)
        super().__init__(None,  # brush
                         wx.ThePenList.FindOrCreatePen(wx.BLACK, line_width),
                         path)
