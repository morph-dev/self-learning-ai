from morphzero.game.base import Player


def get_result_message(winner, player_configs):
    if winner is None:
        raise ValueError("Game is not over yet!")
    elif winner == Player.NO_PLAYER:
        return "It is a draw!"
    else:
        return f"Winner is {player_configs[winner].name}!"


class GameGraphicsContext:
    def __init__(self, game_config, graphics_renderer, player_colors):
        self.game_config = game_config
        self.graphics_renderer = graphics_renderer
        self.player_colors = player_colors
