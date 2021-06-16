import threading
import time

import wx

from morphzero.game.base import Player, GameService
from morphzero.ui.util import matrixgame
from morphzero.ui.common import GameGraphicsContext
from morphzero.ui.util.player_name_decorator import ColorPlayerNameDecorator

_MIN_AI_PLAY_TIME_SEC = 0.2


class BaseGamePanel(wx.Panel, GameService.Listener):
    def __init__(self, game_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.SetDoubleBuffered(True)

        self.game_config = game_config
        self.game_service = GameService(self.game_config.rules.create_game_engine())
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

    def create_layout(self):
        name_decorator = ColorPlayerNameDecorator(self.game_graphics_context)

        def create_player_name_static_text(player):
            player_name = wx.StaticText(self, label=self.game_config.players[player].name)
            name_decorator.decorate_player_label(player, player_name)
            return player_name

        first_player_name, second_player_name = [
            create_player_name_static_text(player)
            for player in [Player.FIRST_PLAYER, Player.SECOND_PLAYER]
        ]
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(first_player_name,
                  wx.SizerFlags().Left().Border())
        sizer.Add(self.board,
                  wx.SizerFlags(1).Expand())
        sizer.Add(second_player_name,
                  wx.SizerFlags().Right().Border())
        self.SetSizerAndFit(sizer)

    def create_board(self):
        raise NotImplementedError()

    def show_result(self, state):
        raise NotImplementedError()

    def maybe_play_ai_move(self):
        """
        Uses separate Thread to make a move for the AI, if it is AI's turn.
        """
        game_service = self.game_service
        state = game_service.state
        ai_player = self.game_config.players[state.current_player].ai_player

        def play_ai_move(min_run_time_sec=_MIN_AI_PLAY_TIME_SEC):
            start_time_sec = time.time()
            ai_move = ai_player.play_move(game_service.engine, state)
            elapsed_time_sec = time.time() - start_time_sec
            if min_run_time_sec > elapsed_time_sec:
                time.sleep(min_run_time_sec - elapsed_time_sec)
            wx.CallAfter(game_service.play_move, ai_move)
            wx.CallAfter(wx.EndBusyCursor)

        if not state.is_game_over and ai_player:
            wx.BeginBusyCursor()
            threading.Thread(target=play_ai_move).start()

    # window events
    def on_destroy(self, _):
        self.game_service.remove_listener(self)

    # GameService events
    def on_new_game(self, state):
        self.board.Refresh()

    def on_move(self, old_state, move, new_state):
        self.board.Refresh()
        self.maybe_play_ai_move()

    def on_game_over(self, state):
        self.show_result(state)


class AiPlayerThread(threading.Thread):
    def __init__(self, ):
        super().__init__()


class BaseHoverDrawer(matrixgame.MatrixGameBoard.AdditionalDrawing):
    def __init__(self, board):
        self.board = board
        self.hover_board_coordinates = None

        self.board.Bind(wx.EVT_PAINT, self.on_paint)
        self.board.Bind(wx.EVT_MOTION, self.on_motion)
        self.board.Bind(wx.EVT_LEAVE_WINDOW, self.on_leave)

    def get_board_coordinates_for_mouse_event(self, event):
        raise NotImplementedError()

    def draw(self, gc):
        raise NotImplementedError()

    def on_paint(self, event):
        dc = wx.PaintDC(self.board)
        gc = wx.GraphicsContext.Create(dc)
        self.draw(gc)
        event.Skip()

    def on_motion(self, event):
        self.hover_board_coordinates = self.get_board_coordinates_for_mouse_event(event)
        self.board.Refresh()

    def on_leave(self, _):
        self.hover_board_coordinates = None
        self.board.Refresh()
