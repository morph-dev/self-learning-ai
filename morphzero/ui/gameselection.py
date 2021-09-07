from __future__ import annotations

from typing import NamedTuple, Callable, Any, Optional, TypeVar, Dict, List

import wx

from morphzero.ai.model import Model
from morphzero.core.game import Player, Rules
from morphzero.ui.gameconfig import GameConfig, PlayerConfig, GameType


class GameConfigParams(NamedTuple):
    """Configuration parameters for setting single game and creating GameConfig."""
    name: str
    type: GameType
    rules: Rules
    player_config_params_list: List[PlayerConfigParams]


class PlayerConfigParams(NamedTuple):
    """Configuration parameters for setting up player and creating PlayerConfig."""
    default_name: str
    ai_model_factory: Optional[Callable[[Rules], Model]]


class GameSelectionDialog(wx.Dialog):
    """The dialog that allows user to select and configure game."""
    panel: GameSelectionPanel

    def __init__(self, game_selection_state: GameSelectionState, **kwargs: Any):
        super().__init__(title="Select game", **kwargs)
        self.panel = GameSelectionPanel(game_selection_state, parent=self)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.panel, wx.SizerFlags(1).Expand().Border())
        sizer.Add(self.CreateButtonSizer(wx.OK), wx.SizerFlags().Expand().Border())

        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.SetSizerAndFit(sizer)

    def get_game_selection_state(self) -> GameSelectionState:
        return self.panel.create_state()

    def on_close(self, _: wx.CloseEvent) -> None:
        self.EndModal(wx.ID_CLOSE)


class GameSelectionState(NamedTuple):
    """The State of the GameSelectionDialog and GameSelectionPanel."""
    game_config_params_list: List[GameConfigParams]
    selected_game_config_params: Optional[GameConfigParams] = None
    player_box_states: Dict[Player, PlayerBoxState] = dict()

    def create_game_config(self) -> GameConfig:
        game_config_params = self.selected_game_config_params
        assert game_config_params and len(self.player_box_states) > 0

        return GameConfig(name=game_config_params.name,
                          type=game_config_params.type,
                          rules=game_config_params.rules,
                          players={
                              player: self.player_box_states[player].create_player_config(game_config_params.rules)
                              for player in [Player.FIRST_PLAYER, Player.SECOND_PLAYER]
                          })

    def swap_players(self) -> None:
        p1 = Player.FIRST_PLAYER
        p2 = Player.SECOND_PLAYER
        self.player_box_states[p1], self.player_box_states[p2] = \
            self.player_box_states[p2], self.player_box_states[p1]


class GameSelectionPanel(wx.Panel):
    """The Ui component for selecting and configuring Game."""
    state: GameSelectionState
    rules_choice: wx.Choice
    player_boxes: Dict[Player, PlayerBox]

    def __init__(self, state: GameSelectionState, **kwargs: Any):
        super().__init__(**kwargs)
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
            Player.FIRST_PLAYER: PlayerBox(rules_choice=self.rules_choice,
                                           player_box_state=self.state.player_box_states.get(Player.FIRST_PLAYER),
                                           parent=self,
                                           label="First player"),
            Player.SECOND_PLAYER: PlayerBox(rules_choice=self.rules_choice,
                                            player_box_state=self.state.player_box_states.get(Player.SECOND_PLAYER),
                                            parent=self,
                                            label="Second player")
        }
        for player_box in self.player_boxes.values():
            sizer.Add(player_box.sizer, wx.SizerFlags().Expand().Border())

        self.SetSizer(sizer)

    def create_rules_choice(self) -> wx.Choice:
        choice = wx.Choice(self)
        _populate_choice_with_params(choice,
                                     self.state.game_config_params_list,
                                     lambda params: params.name,
                                     self.state.selected_game_config_params)
        return choice

    def create_state(self) -> GameSelectionState:
        game_config_params = self.rules_choice.GetClientData(self.rules_choice.GetSelection())
        return GameSelectionState(
            game_config_params_list=self.state.game_config_params_list,
            selected_game_config_params=game_config_params,
            player_box_states={
                player: self.player_boxes[player].create_player_box_state()
                for player in [Player.FIRST_PLAYER, Player.SECOND_PLAYER]
            })


class PlayerBoxState(NamedTuple):
    """State of the PlayerBox class."""
    selected_player_config_params: PlayerConfigParams
    selected_name: str

    def create_player_config(self, rules: Rules) -> PlayerConfig:
        ai_model_factory = self.selected_player_config_params.ai_model_factory
        return PlayerConfig(name=self.selected_name,
                            ai_model=ai_model_factory(rules) if ai_model_factory else None)


class PlayerBox(wx.StaticBox):
    """The Ui component for configuring the Player."""
    rules_choice: wx.Choice
    ai_model_choice: wx.Choice
    name: wx.TextCtrl

    def __init__(self,
                 rules_choice: wx.Choice,
                 player_box_state: Optional[PlayerBoxState],
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.rules_choice = rules_choice
        self.rules_choice.Bind(wx.EVT_CHOICE, self.on_rules_choice)

        grid_sizer = wx.GridBagSizer(vgap=5, hgap=5)
        # row 0
        grid_sizer.Add(wx.StaticText(self, label="Type:"), pos=(0, 0), flag=wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        self.ai_model_choice = self.create_ai_model_choice(player_box_state)
        grid_sizer.Add(self.ai_model_choice, pos=(0, 1), flag=wx.EXPAND)
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

    def get_game_config_params(self) -> GameConfigParams:
        game_config_params = self.rules_choice.GetClientData(self.rules_choice.GetSelection())
        assert isinstance(game_config_params, GameConfigParams)
        return game_config_params

    def create_ai_model_choice(self, player_box_state: Optional[PlayerBoxState]) -> wx.Choice:
        ai_model_choice = wx.Choice(self)
        ai_model_choice.Bind(wx.EVT_CHOICE, self.on_ai_model_choice)

        game_config_params = self.get_game_config_params()
        _populate_choice_with_params(ai_model_choice,
                                     game_config_params.player_config_params_list,
                                     lambda params: params.default_name,
                                     player_box_state.selected_player_config_params if player_box_state else None)

        return ai_model_choice

    def update_ai_model_choice(self) -> None:
        self.ai_model_choice.Clear()

        game_config_params = self.get_game_config_params()
        _populate_choice_with_params(self.ai_model_choice,
                                     game_config_params.player_config_params_list,
                                     lambda params: params.default_name)
        self.GetTopLevelParent().Fit()

    def update_name(self) -> None:
        self.name.SetValue(self.get_selected_player_config_params().default_name)

    def on_rules_choice(self, event: wx.CommandEvent) -> None:
        self.update_ai_model_choice()
        self.update_name()
        event.Skip()

    def on_ai_model_choice(self, _: wx.CommandEvent) -> None:
        self.update_name()

    def get_selected_player_config_params(self) -> PlayerConfigParams:
        player_config_params = self.ai_model_choice.GetClientData(self.ai_model_choice.GetSelection())
        assert isinstance(player_config_params, PlayerConfigParams)
        return player_config_params

    def create_player_box_state(self) -> PlayerBoxState:
        return PlayerBoxState(
            selected_player_config_params=self.get_selected_player_config_params(),
            selected_name=self.name.GetValue())


T = TypeVar("T")


def _populate_choice_with_params(choice: wx.Choice,
                                 params_list: List[T],
                                 name_function: Callable[[T], str],
                                 selected_params: Optional[T] = None) -> None:
    for params in params_list:
        item_index = choice.Append(name_function(params), params)
        if selected_params == params:
            choice.SetSelection(item_index)
    if choice.GetSelection() == wx.NOT_FOUND:
        choice.SetSelection(0)
