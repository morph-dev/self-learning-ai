class PlayerNameDecorator:
    def decorate_player_label(self, player, wx_static_text):
        raise NotImplementedError()


class ColorPlayerNameDecorator(PlayerNameDecorator):
    def __init__(self, game_graphics_context):
        self.game_graphics_context = game_graphics_context

    def decorate_player_label(self, player, wx_static_text):
        wx_static_text.SetForegroundColour(self.game_graphics_context.player_colors[player])
