from collections import namedtuple

import wx

from morphzero.game.base import Player
from morphzero.ui.gameconfig import GameConfig, PlayerConfig

GameConfigParams = namedtuple("GameConfigParams",
                              ["name", "type", "rules", "player_config_params_list"])

PlayerConfigParams = namedtuple("PlayerConfigParams",
                                ["default_name", "factory"])


class GameSelectionState(namedtuple("GameSelectionState",
                                    [
                                        "game_config_params_list",
                                        "selected_game_config_params",
                                        "player_box_states"
                                    ],
                                    defaults=[None, dict()])):
    def create_game_config(self):
        game_config_params = self.selected_game_config_params
        return GameConfig(name=game_config_params.name,
                          type=game_config_params.type,
                          rules=game_config_params.rules,
                          players={
                              player: self.player_box_states[player].create_player_config()
                              for player in [Player.FIRST_PLAYER, Player.SECOND_PLAYER]
                          })

    def swap_players(self):
        p1 = Player.FIRST_PLAYER
        p2 = Player.SECOND_PLAYER
        self.player_box_states[p1], self.player_box_states[p2] = \
            self.player_box_states[p2], self.player_box_states[p1]


class PlayerBoxState(namedtuple("PlayerBoxState",
                                ["selected_player_config_params", "selected_name"],
                                defaults=[None, ""])):
    def create_player_config(self):
        factory = self.selected_player_config_params.factory
        return PlayerConfig(name=self.selected_name,
                            ai_player=factory() if factory else None)


class GameSelectionDialog(wx.Dialog):
    def __init__(self, game_selection_state, *args, **kw):
        super().__init__(title="Select game", *args, **kw)
        self.panel = GameSelectionPanel(game_selection_state, parent=self)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.panel, wx.SizerFlags(1).Expand().Border())
        sizer.Add(self.CreateButtonSizer(wx.OK), wx.SizerFlags().Expand().Border())

        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.SetSizerAndFit(sizer)

    def get_game_selection_state(self):
        return self.panel.create_state()

    def on_close(self, _):
        self.EndModal(wx.ID_CLOSE)


class GameSelectionPanel(wx.Panel):
    def __init__(self, state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state

        sizer = wx.BoxSizer(wx.VERTICAL)

        rules_sizer = wx.BoxSizer(wx.HORIZONTAL)
        rules_sizer.Add(wx.StaticText(self, label="Rules:"),
                        wx.SizerFlags().Center().Border())
        self.rules_choice = self.create_rules_choice()
        rules_sizer.Add(self.rules_choice,
                        wx.SizerFlags(1).Center().Border())
        sizer.Add(rules_sizer, wx.SizerFlags().Expand())

        self.player_boxes = {
            Player.FIRST_PLAYER: PlayerBox(self.rules_choice,
                                           self.state.player_box_states.get(Player.FIRST_PLAYER),
                                           parent=self,
                                           label="First player"),
            Player.SECOND_PLAYER: PlayerBox(self.rules_choice,
                                            self.state.player_box_states.get(Player.SECOND_PLAYER),
                                            parent=self,
                                            label="Second player")
        }
        for player_box in self.player_boxes.values():
            sizer.Add(player_box.sizer, wx.SizerFlags().Expand().Border())

        self.SetSizer(sizer)

    def create_rules_choice(self):
        choice = wx.Choice(self)
        _populate_choice_with_params(choice,
                                     self.state.game_config_params_list,
                                     lambda params: params.name,
                                     self.state.selected_game_config_params)
        return choice

    def create_state(self):
        return GameSelectionState(
            game_config_params_list=self.state.game_config_params_list,
            selected_game_config_params=self.rules_choice.GetClientData(self.rules_choice.GetSelection()),
            player_box_states={
                player: self.player_boxes[player].create_player_box_state()
                for player in [Player.FIRST_PLAYER, Player.SECOND_PLAYER]
            })


class PlayerBox(wx.StaticBox):
    def __init__(self, rules_choice, player_box_state, *args, **kw):
        super().__init__(*args, **kw)
        self.rules_choice = rules_choice
        self.rules_choice.Bind(wx.EVT_CHOICE, self.on_rules_choice)

        grid_sizer = wx.GridBagSizer(vgap=5, hgap=5)
        # row 0
        grid_sizer.Add(wx.StaticText(self, label="Type:"), pos=(0, 0), flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        self.ai_player_choice = self.create_ai_player_choice(player_box_state)
        grid_sizer.Add(self.ai_player_choice, pos=(0, 1), flag=wx.EXPAND)
        # row 1
        grid_sizer.Add(wx.StaticText(self, label="Name:"), pos=(1, 0), flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        self.name = wx.TextCtrl(self)
        if player_box_state and player_box_state.selected_name:
            self.name.SetValue(player_box_state.selected_name)
        else:
            self.update_name()
        grid_sizer.Add(self.name, pos=(1, 1), flag=wx.EXPAND)

        grid_sizer.AddGrowableCol(1, 1)

        self.sizer = wx.StaticBoxSizer(self, wx.VERTICAL)
        self.sizer.Add(grid_sizer, wx.SizerFlags(1).Expand().Border())

    def create_ai_player_choice(self, player_box_state):
        ai_player_choice = wx.Choice(self)
        ai_player_choice.Bind(wx.EVT_CHOICE, self.on_ai_player_choice)

        game_config_params = self.rules_choice.GetClientData(self.rules_choice.GetSelection())
        _populate_choice_with_params(ai_player_choice,
                                     game_config_params.player_config_params_list,
                                     lambda params: params.default_name,
                                     player_box_state.selected_player_config_params if player_box_state else None)

        return ai_player_choice

    def update_ai_player_choice(self):
        self.ai_player_choice.Clear()

        game_config_params = self.rules_choice.GetClientData(self.rules_choice.GetSelection())
        _populate_choice_with_params(self.ai_player_choice,
                                     game_config_params.player_config_params_list,
                                     lambda params: params.default_name)
        self.GetTopLevelParent().Fit()

    def update_name(self):
        self.name.SetValue(self.get_selected_player_config_params().default_name)

    def on_rules_choice(self, event):
        self.update_ai_player_choice()
        self.update_name()
        event.Skip()

    def on_ai_player_choice(self, _):
        self.update_name()

    def get_selected_player_config_params(self):
        return self.ai_player_choice.GetClientData(self.ai_player_choice.GetSelection())

    def create_player_box_state(self):
        return PlayerBoxState(
            selected_player_config_params=self.get_selected_player_config_params(),
            selected_name=self.name.GetValue())


def _populate_choice_with_params(choice, params_list, name_function, selected_params=None):
    for params in params_list:
        item_index = choice.Append(name_function(params), params)
        if selected_params == params:
            choice.SetSelection(item_index)
    if choice.GetSelection() == wx.NOT_FOUND:
        choice.SetSelection(0)
