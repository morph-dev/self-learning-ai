from dataclasses import dataclass
from typing import Optional

import wx

from morphzero.core.game import Player
from morphzero.ui.gameconfig import PlayerConfig, GameConfig


def get_result_message(winner: Optional[Player], player_configs: dict[Player, PlayerConfig]) -> str:
    if winner is None:
        raise ValueError("Game is not over yet!")
    elif winner == Player.NO_PLAYER:
        return "It is a draw!"
    else:
        return f"Winner is {player_configs[winner].name}!"


@dataclass(frozen=True)
class GameGraphicsContext:
    game_config: GameConfig
    graphics_renderer: wx.GraphicsRenderer
    player_colors: dict[Player, wx.Colour]
