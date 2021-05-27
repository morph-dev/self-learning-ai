import tkinter as tk
from collections import namedtuple
from tkinter import ttk

from ui.gameconfig import GameConfig

RulesConfig = namedtuple("RulesConfig",
                         ["name", "rules", "models_config"])

ModelConfig = namedtuple("ModelConfig", ["name", "path", "constructor"])


def _update_combobox_values(combobox, configs):
    def same_lists(list1, list2):
        if len(list1) != len(list2):
            return False
        return all(a == b for a, b in zip(list1, list2))

    new_values = [config.name for config in configs]
    if not same_lists(combobox["values"], new_values):
        combobox["values"] = new_values
        combobox.current(0)


class GameConfigApp(tk.Tk):
    def __init__(self, rules_configs):
        if not rules_configs:
            raise ValueError("At least one rule required.")
        super().__init__()
        self.game_config = None

        self.title("Game Config")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        _Frame(self, rules_configs).grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)


class _Frame(ttk.Frame):
    def __init__(self, container, rules_configs):
        super().__init__(container)
        self.rules_configs = rules_configs

        self.columnconfigure(1, weight=1)

        # ROW 0
        ttk.Label(self, text="Game").grid(row=0, column=0, padx=5, pady=5)

        self.rules_combobox_text = tk.StringVar()
        self.rules_combobox = ttk.Combobox(self, textvariable=self.rules_combobox_text, state="readonly")
        self.rules_combobox.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        # ROW 1
        self.first_player_frame = _PlayerFrame(self, "First player", self.rules_combobox_text)
        self.first_player_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        # ROW 2
        self.second_player_frame = _PlayerFrame(self, "Second player", self.rules_combobox_text)
        self.second_player_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        # ROW 3
        start_button = ttk.Button(self, text="Start", command=self.on_start_clicked)
        start_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        _update_combobox_values(self.rules_combobox, self.rules_configs)

    @property
    def rules_config(self):
        return self.rules_configs[self.rules_combobox.current()]

    def on_start_clicked(self):
        rules_config = self.rules_config
        models_config = rules_config.models_config

        def create_model(player_frame):
            model_config = models_config[player_frame.player_combobox.current()]
            return model_config.constructor(model_config.path) if model_config.constructor else None

        self.winfo_toplevel().game_config = GameConfig(
            rules=rules_config.rules,
            first_player_name=self.first_player_frame.name.get(),
            first_player_model=create_model(self.first_player_frame),
            second_player_name=self.second_player_frame.name.get(),
            second_player_model=create_model(self.second_player_frame))
        self.winfo_toplevel().destroy()


class _PlayerFrame(ttk.LabelFrame):
    def __init__(self, container, text, rules_combobox_text):
        super().__init__(container, text=text)
        rules_combobox_text.trace_add("write", self.on_rules_config_change)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        inner_frame = ttk.Frame(self)
        inner_frame.grid(row=0, column=0, padx=2, pady=2, sticky=tk.NSEW)
        inner_frame.columnconfigure(1, weight=1)

        self.combobox_text = tk.StringVar()
        self.combobox_text.trace_add("write", self.on_combobox_change)
        self.player_combobox = ttk.Combobox(inner_frame, state="readonly", textvariable=self.combobox_text)
        self.player_combobox.grid(row=0, column=0, columnspan=2, padx=2, pady=2, sticky=tk.EW)

        ttk.Label(inner_frame, text="name:").grid(row=1, column=0, padx=2, pady=2)
        self.name = tk.StringVar()
        ttk.Entry(inner_frame, textvariable=self.name).grid(row=1, column=1, padx=2, pady=2, sticky=tk.EW)

    def on_rules_config_change(self, *_):
        rules_config = self.master.rules_config
        models_config = rules_config.models_config
        _update_combobox_values(self.player_combobox, models_config)

    def on_combobox_change(self, *_):
        self.name.set(self.player_combobox.get())
