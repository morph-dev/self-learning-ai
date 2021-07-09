from typing import Any, Optional

import wx

from morphzero.ui.basegamepanel import BaseGamePanel
from morphzero.ui.gameconfig import GameType, GameConfig
from morphzero.ui.gameselection import GameSelectionDialog, GameSelectionState


class GameApp(wx.App):
    """The main app for playing the game."""

    def __init__(self, game_selection_state: GameSelectionState):
        self.game_selection_state = game_selection_state
        super().__init__()

    def OnInit(self) -> bool:
        super().OnInit()
        # Let user choice a game.
        if self.open_select_game_dialog():
            game_frame = GameFrame(self)
            self.SetTopWindow(game_frame)
            game_frame.Show()
        return True

    def open_select_game_dialog(self) -> bool:
        with GameSelectionDialog(self.game_selection_state, parent=None) as dialog:
            if dialog.ShowModal() == wx.ID_OK:
                self.game_selection_state = dialog.panel.create_state()
                return True
            else:
                return False


class GameFrame(wx.Frame):
    game_app: GameApp
    game_panel: Optional[BaseGamePanel]

    def __init__(self, game_app: GameApp, **kwargs: Any):
        self.game_app = game_app
        super().__init__(parent=None,
                         style=wx.DEFAULT_FRAME_STYLE & ~(wx.RESIZE_BORDER | wx.MAXIMIZE_BOX),
                         **kwargs)

        self.init_menu_bar()

        self.game_panel = None
        self.init_game()

    def init_menu_bar(self) -> None:
        game_menu = wx.Menu()

        change_game_item = game_menu.Append(wx.ID_ANY, "&Change game\tCtrl+C")
        self.Bind(wx.EVT_MENU, self.on_change_game, change_game_item)

        restart_game_item = game_menu.Append(wx.ID_ANY, "&Restart Game\tCtrl+R")
        self.Bind(wx.EVT_MENU, self.on_restart_game, restart_game_item)

        change_sides_item = game_menu.Append(wx.ID_ANY, "Change &sides and restart\tCtrl+S")
        self.Bind(wx.EVT_MENU, self.on_change_sides, change_sides_item)

        menu_bar = wx.MenuBar()
        menu_bar.Append(game_menu, "&Game")
        self.SetMenuBar(menu_bar)

    def on_change_game(self, _: wx.CommandEvent) -> None:
        if self.game_app.open_select_game_dialog():
            if self.game_panel:
                self.game_panel.Destroy()
            self.init_game()

    def on_restart_game(self, _: wx.CommandEvent) -> None:
        if self.game_panel:
            self.game_panel.Destroy()
        self.init_game()

    def on_change_sides(self, _: wx.CommandEvent) -> None:
        self.game_app.game_selection_state.swap_players()
        if self.game_panel:
            self.game_panel.Destroy()
        self.init_game()

    def init_game(self) -> None:
        game_config = self.game_app.game_selection_state.create_game_config()
        self.SetTitle(game_config.name)

        self.game_panel = self.create_game_panel(game_config)

        sizer = wx.BoxSizer()
        sizer.Add(self.game_panel, wx.SizerFlags(1).Expand())
        self.SetSizerAndFit(sizer)

    def create_game_panel(self, game_config: GameConfig) -> BaseGamePanel:
        if game_config.type == GameType.TIC_TAC_TOE or \
                game_config.type == GameType.GOMOKU:
            from morphzero.games.genericgomoku.ui.panel import GenericGomokuPanel
            return GenericGomokuPanel(game_config=game_config, parent=self)

        elif game_config.type == GameType.CONNECT_FOUR:
            from morphzero.games.connectfour.ui.panel import ConnectFourPanel
            return ConnectFourPanel(game_config=game_config, parent=self)

        else:
            raise ValueError(f"Unexpected GameType: {game_config.type}")
