import wx

from morphzero.core.game import Player
from morphzero.ui.common import GameGraphicsContext


class PlayerNameDecorator:
    def decorate_player_label(self, player: Player, player_label: wx.StaticText) -> None:
        raise NotImplementedError()


class ColorPlayerNameDecorator(PlayerNameDecorator):
    game_graphics_context: GameGraphicsContext

    def __init__(self, game_graphics_context: GameGraphicsContext):
        self.game_graphics_context = game_graphics_context

    def decorate_player_label(self, player: Player, player_label: wx.StaticText) -> None:
        player_label.SetForegroundColour(self.game_graphics_context.player_colors[player])
