from __future__ import annotations

from typing import Optional, Any

import wx

from morphzero.core.common.matrix_board import MatrixBoardCoordinates
from morphzero.core.game import Player, State
from morphzero.games.connectfour.game import ConnectFourMove, ConnectFourEngine, ConnectFourState
from morphzero.ui.basegamepanel import BaseGamePanel, BaseHoverDrawer
from morphzero.ui.common import get_result_message
from morphzero.ui.util import matrixgame
from morphzero.ui.util.matrixgame import MatrixGameBoard
from morphzero.ui.util.painter import CirclePainter, Painter


class ConnectFourPanel(BaseGamePanel, MatrixGameBoard.OnClickCallback):
    board: MatrixGameBoard
    hover_drawer: HoverDrawer

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.hover_drawer = HoverDrawer(self)

    def create_board(self) -> MatrixGameBoard:
        painters = self.create_painters()
        return matrixgame.MatrixGameBoard(parent=self,
                                          game_service=self.game_service,
                                          on_click_callback=self,
                                          grid_painter=matrixgame.GridPainter(self.game_graphics_context),
                                          painters=painters)

    def create_painters(self) -> dict[Player, Painter]:
        return {
            player: CirclePainter(
                self.game_graphics_context,
                self.game_graphics_context.player_colors[player])
            for player in [Player.FIRST_PLAYER, Player.SECOND_PLAYER]
        }

    def show_result(self, state: State) -> None:
        assert state.is_game_over and state.result
        result_message = get_result_message(state.result.winner, self.game_config.players)
        wx.MessageDialog(self, result_message).ShowModal()

    def on_click(self, board_coordinates: MatrixBoardCoordinates) -> None:
        current_player = self.game_service.state.current_player
        if not self.game_config.players[current_player].ai_model:
            move = self.get_playable_move_for_board_coordinates(board_coordinates)
            if move:
                self.game_service.play_move(move)

    def get_playable_move_for_board_coordinates(
            self, board_coordinates: MatrixBoardCoordinates) -> Optional[ConnectFourMove]:
        """Finds a playable move (if it exists) that matches the column."""
        engine = self.game_service.engine
        state = self.game_service.state
        assert isinstance(engine, ConnectFourEngine) and isinstance(state, ConnectFourState)

        for move in engine.playable_moves(state):
            if move.coordinates and (move.coordinates.column == board_coordinates.column):
                return move
        return None


class HoverDrawer(BaseHoverDrawer):
    board: MatrixGameBoard
    hover_board_coordinates: Optional[MatrixBoardCoordinates]
    panel: ConnectFourPanel

    def __init__(self, panel: ConnectFourPanel):
        self.panel = panel
        super().__init__(self.panel.board)

    def get_board_coordinates_for_mouse_event(self, event: wx.MouseEvent) -> MatrixBoardCoordinates:
        dc = wx.ClientDC(self.board)
        return self.board.mouse_position_to_board_coordinates(
            event.GetLogicalPosition(dc))

    def draw(self, gc: wx.GraphicsContext) -> None:
        if self.hover_board_coordinates is None:
            # we don't have anything to draw
            return

        state = self.panel.game_service.state
        if state.is_game_over:
            # don't draw if it is game over
            return

        if self.panel.game_config.players[state.current_player].ai_model:
            # don't draw if it is computers turn
            return

        move = self.panel.get_playable_move_for_board_coordinates(self.hover_board_coordinates)
        painter = self.board.painters.get(state.current_player)

        if move and painter:
            assert move.coordinates

            transformation = gc.CreateMatrix()
            transformation.Scale(*self.board.get_cell_size())
            transformation.Translate(move.coordinates.column, move.coordinates.row)

            gc.BeginLayer(0.3)
            painter.transform_and_paint(gc, transformation)
            gc.EndLayer()
