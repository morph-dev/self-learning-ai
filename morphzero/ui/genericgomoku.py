import wx

from morphzero.game.base import Player
from morphzero.game.genericgomoku import GenericGomokuMove
from morphzero.ui.util import matrixgame
from morphzero.ui.basegamepanel import BaseGamePanel, BaseHoverDrawer
from morphzero.ui.common import get_result_message
from morphzero.ui.gameconfig import GameType
from morphzero.ui.util.painter import SimplePainter, CirclePainter


class GenericGomokuPanel(BaseGamePanel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hover_drawer = self.HoverDrawer(self)

    def create_board(self):
        painters = self.create_painters()
        return matrixgame.MatrixGameBoard(parent=self,
                                          game_service=self.game_service,
                                          on_click_callback=self.on_click,
                                          grid_painter=matrixgame.GridPainter(self.game_graphics_context),
                                          painters=painters)

    def create_painters(self):
        if self.game_config.type == GameType.TIC_TAC_TOE:
            return {
                player: painter(self.game_graphics_context, self.game_graphics_context.player_colors[player])
                for player, painter in zip([Player.FIRST_PLAYER, Player.SECOND_PLAYER],
                                           [XPainter, OPainter])
            }
        else:
            return {
                player: CirclePainter(
                    self.game_graphics_context,
                    self.game_graphics_context.player_colors[player])
                for player in [Player.FIRST_PLAYER, Player.SECOND_PLAYER]
            }

    def show_result(self, state):
        result_message = get_result_message(state.result.winner, self.game_config.players)
        wx.MessageDialog(self, result_message).ShowModal()

    def on_click(self, board_coordinates):
        state = self.game_service.state
        current_player = state.current_player
        if not self.game_config.players[current_player].ai_player:
            move = GenericGomokuMove(*board_coordinates)
            if self.game_service.is_move_playable(move):
                self.game_service.play_move(move)

    class HoverDrawer(BaseHoverDrawer):
        def __init__(self, panel):
            self.panel = panel
            super().__init__(self.panel.board)

        def get_board_coordinates_for_mouse_event(self, event):
            dc = wx.ClientDC(self.board)
            return self.board.mouse_position_to_board_coordinates(
                event.GetLogicalPosition(dc))

        def draw(self, gc):
            if self.hover_board_coordinates is None:
                # we don't have anything to draw
                return

            state = self.panel.game_service.state
            current_player = state.current_player
            if self.panel.game_config.players[current_player].ai_player:
                # don't draw if it is computers turn
                return

            if not self.panel.game_service.is_move_playable(
                    GenericGomokuMove(*self.hover_board_coordinates)):
                # move not possible
                return

            row, column = self.hover_board_coordinates
            painter = self.board.painters.get(current_player)

            transformation = gc.CreateMatrix()
            transformation.Scale(*self.board.get_cell_size())
            transformation.Translate(column, row)

            gc.BeginLayer(0.3)
            painter.transform_and_paint(gc, transformation)
            gc.EndLayer()


class XPainter(SimplePainter):
    def __init__(self, game_graphics_context, color, line_width=5):
        path = game_graphics_context.graphics_renderer.CreatePath()
        path.MoveToPoint(0.1, 0.1)
        path.AddLineToPoint(0.9, 0.9)
        path.MoveToPoint(0.1, 0.9)
        path.AddLineToPoint(0.9, 0.1)
        super().__init__(None,
                         wx.ThePenList.FindOrCreatePen(color, line_width),
                         path)


class OPainter(SimplePainter):
    def __init__(self, game_graphics_context, color, line_width=5):
        path = game_graphics_context.graphics_renderer.CreatePath()
        path.AddEllipse(0.1, 0.1, 0.8, 0.8)
        super().__init__(None,
                         wx.ThePenList.FindOrCreatePen(color, line_width),
                         path)
