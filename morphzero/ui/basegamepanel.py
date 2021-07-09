import threading
import time
from collections import Callable
from typing import Optional, Any, TypeVar

import wx

from morphzero.core.common.matrix_board import MatrixBoardCoordinates
from morphzero.core.game import Player, State, Move
from morphzero.core.game_service import GameService, GameServiceListener
from morphzero.ui.common import GameGraphicsContext
from morphzero.ui.gameconfig import GameConfig
from morphzero.ui.util import matrixgame
from morphzero.ui.util.player_name_decorator import ColorPlayerNameDecorator

_MIN_AI_PLAY_TIME_SEC: float = 0.2

T = TypeVar('T')


def _execute_off_thread(
        function: Callable[[], T],
        callback: Callable[[T], None],
        use_busy_cursor: bool = False,
        min_duration: Optional[float] = None) -> None:
    def off_thread() -> None:
        if use_busy_cursor:
            wx.CallAfter(wx.BeginBusyCursor)
        start_time_sec = time.time()
        t = function()
        elapsed_time_sec = time.time() - start_time_sec
        if min_duration and min_duration > elapsed_time_sec:
            time.sleep(min_duration - elapsed_time_sec)
        if use_busy_cursor:
            wx.CallAfter(wx.EndBusyCursor)
        wx.CallAfter(main_thread_callback, t)

    threading.Thread(target=off_thread).start()

    def main_thread_callback(t: T) -> None:
        if use_busy_cursor:
            wx.EndBusyCursor()
        callback(t)


class BaseGamePanel(wx.Panel, GameServiceListener):
    game_config: GameConfig
    game_service: GameService
    game_graphics_context: GameGraphicsContext
    board: wx.Window

    def __init__(self, game_config: GameConfig, **kwargs: Any):
        super().__init__(**kwargs)
        self.SetDoubleBuffered(True)

        self.game_config = game_config
        self.game_service = GameService(self.game_config.rules.create_engine())
        self.game_graphics_context = GameGraphicsContext(
            game_config=self.game_config,
            graphics_renderer=wx.GraphicsRenderer.GetDefaultRenderer(),
            player_colors={
                Player.FIRST_PLAYER: wx.BLUE,
                Player.SECOND_PLAYER: wx.RED,
            })
        self.board = self.create_board()

        # create layout
        self.create_layout()

        # bind
        self.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)

        # Init game service
        self.game_service.add_listener(self)
        self.game_service.new_game()
        self.maybe_play_ai_move()

    def create_layout(self) -> None:
        name_decorator = ColorPlayerNameDecorator(self.game_graphics_context)

        def create_player_name_static_text(player: Player) -> wx.StaticText:
            player_name = wx.StaticText(self, label=self.game_config.players[player].name)
            name_decorator.decorate_player_label(player, player_name)
            return player_name

        first_player_name, second_player_name = (
            create_player_name_static_text(player)
            for player in [Player.FIRST_PLAYER, Player.SECOND_PLAYER]
        )
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(first_player_name,
                  wx.SizerFlags().Left().Border())
        sizer.Add(self.board,
                  wx.SizerFlags(1).Expand())
        sizer.Add(second_player_name,
                  wx.SizerFlags().Right().Border())
        self.SetSizerAndFit(sizer)

    def create_board(self) -> wx.Window:
        raise NotImplementedError()

    def show_result(self, state: State) -> None:
        raise NotImplementedError()

    def maybe_play_ai_move(self) -> None:
        """Uses separate Thread to make a move for the AI, if it is AI's turn."""
        game_service = self.game_service
        state = game_service.state
        if state.is_game_over:
            return
        ai_model = self.game_config.players[state.current_player].ai_model

        if ai_model:
            def play() -> Move:
                assert ai_model
                move_or_move_index = ai_model.play_move(state)
                if isinstance(move_or_move_index, Move):
                    return move_or_move_index
                else:
                    return game_service.engine.create_move_from_move_index(move_or_move_index)

            _execute_off_thread(
                function=play,
                callback=game_service.play_move,
                use_busy_cursor=True,
                min_duration=_MIN_AI_PLAY_TIME_SEC,
            )

    # window events
    def on_destroy(self, _: wx.WindowDestroyEvent) -> None:
        self.game_service.remove_listener(self)

    # GameService events
    def on_new_game(self, state: State) -> None:
        self.board.Refresh()

    def on_move(self, old_state: State, move: Move, new_state: State) -> None:
        self.board.Refresh()
        self.maybe_play_ai_move()

    def on_game_over(self, state: State) -> None:
        self.show_result(state)


class BaseHoverDrawer(matrixgame.MatrixGameBoard.AdditionalDrawing):
    board: wx.Window
    hover_board_coordinates = Optional[MatrixBoardCoordinates]

    def __init__(self, board: wx.Window):
        self.board = board
        self.hover_board_coordinates = None

        self.board.Bind(wx.EVT_PAINT, self.on_paint)
        self.board.Bind(wx.EVT_MOTION, self.on_motion)
        self.board.Bind(wx.EVT_LEAVE_WINDOW, self.on_leave)

    def get_board_coordinates_for_mouse_event(self, event: wx.MouseEvent) -> MatrixBoardCoordinates:
        raise NotImplementedError()

    def draw(self, gc: wx.GraphicsContext) -> None:
        raise NotImplementedError()

    def on_paint(self, event: wx.PaintEvent) -> None:
        dc = wx.PaintDC(self.board)
        gc = wx.GraphicsContext.Create(dc)
        self.draw(gc)
        event.Skip()

    def on_motion(self, event: wx.MouseEvent) -> None:
        self.hover_board_coordinates = self.get_board_coordinates_for_mouse_event(event)
        self.board.Refresh()

    def on_leave(self, _: wx.MouseEvent) -> None:
        self.hover_board_coordinates = None
        self.board.Refresh()
