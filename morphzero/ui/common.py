from dataclasses import dataclass
from typing import Dict

import wx

from morphzero.core.game import Player, Result
from morphzero.ui.gameconfig import PlayerConfig, GameConfig


def get_result_message(result: Result, player_configs: Dict[Player, PlayerConfig]) -> str:
    if result.is_draw:
        return "It is a draw!"
    else:
        return f"Winner is {player_configs[result.winner].name}!"


@dataclass(frozen=True)
class GameGraphicsContext:
    game_config: GameConfig
    graphics_renderer: wx.GraphicsRenderer
    player_colors: Dict[Player, wx.Colour]
