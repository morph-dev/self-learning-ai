import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo

from morphzero.game.base import Player
from morphzero.game.genericgomoku import GenericGomokuGameEngine
from ui.common import get_result_message


def _get_player_color(player):
    if player == Player.FIRST_PLAYER:
        return "blue"
    elif player == Player.SECOND_PLAYER:
        return "red"
    else:
        raise ValueError(f"The {player} doesn't have assigned color.")


class GenericGomokuApp(tk.Tk):
    def __init__(self, game_config):
        super().__init__()
        self.title("Gomoku")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.resizable(width=False, height=False)

        _Frame(self, game_config).grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)


class _Frame(ttk.Frame):
    def __init__(self, container, game_config):
        super().__init__(container)
        self.game_config = game_config
        self.engine = GenericGomokuGameEngine(self.game_config.rules)
        self.state = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        font = "Calibri", 16, "bold"

        ttk.Button(
            self,
            text="New game",
            command=self.on_new_game
        ).grid(row=0, column=0, padx=5, pady=5)

        ttk.Label(
            self,
            text=self.game_config.players[Player.FIRST_PLAYER].name,
            foreground=_get_player_color(Player.FIRST_PLAYER),
            font=font,
        ).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        self.canvas = _Canvas(self, self.engine.rules, self.on_user_play)
        self.canvas.grid(row=2, column=0, padx=5, pady=5, sticky=tk.NSEW)

        ttk.Label(
            self,
            text=self.game_config.players[Player.SECOND_PLAYER].name,
            foreground=_get_player_color(Player.SECOND_PLAYER),
            font=font,
        ).grid(row=3, column=0, padx=5, pady=5, sticky=tk.E)

        self.on_new_game()

    @property
    def is_waiting_for_user_move(self):
        """
        Returns True if engine is waiting for user input, otherwise False (AI, game over).
        """
        if self.state.is_game_over:
            return False
        return self.game_config.players[self.state.current_player].model is None

    @property
    def is_waiting_for_model_move(self):
        """
        Returns True if engine is waiting for AI input, otherwise False (user, game over).
        """
        if self.state.is_game_over:
            return False
        return self.game_config.players[self.state.current_player].model is not None

    def on_new_game(self, *_):
        self.state = self.engine.new_game()
        self.state_updated()

    def on_user_play(self, row, column):
        if not self.is_waiting_for_user_move:
            return
        self.play_move((row, column))

    def play_model_move(self):
        model = self.game_config.players[self.state.current_player].model
        move = model.play_move(self.engine, self.state)
        self.play_move(move)

    def play_move(self, move):
        if self.engine.is_move_playable(self.state, move):
            self.state = self.engine.play_move(self.state, move)
            self.state_updated()

    def state_updated(self):
        self.canvas.state_updated(self.state)
        self.update()
        if self.state.is_game_over:
            showinfo("Game over", get_result_message(self.state, self.game_config.players))
        elif self.is_waiting_for_model_move:
            self.after(500, self.play_model_move)


class _Canvas(tk.Canvas):
    def __init__(self, container, rules, callback):
        desired_width, desired_height = _Canvas._get_desired_size(rules)
        super().__init__(container, width=desired_width, height=desired_height)
        self.rules = rules
        self.callback = callback
        self.state = None

        self.bind("<Configure>", self.on_resize)
        self.bind("<Button-1>", self.on_click)
        self.bind("<Motion>", self.on_motion)
        self.bind("<Leave>", self.on_leave)

    @staticmethod
    def _get_desired_size(rules):
        cell_size = 100 if max(rules.board_size) < 7 else 50
        return rules.board_size[1] * cell_size, rules.board_size[0] * cell_size

    def _get_cell_size(self):
        rows, columns = self.rules.board_size
        return self.winfo_width() / columns, self.winfo_height() / rows

    def _get_row_column_for_event(self, event):
        cell_width, cell_height = self._get_cell_size()
        return int(event.y / cell_height), int(event.x / cell_width)

    def _get_center(self, row, column):
        cell_width, cell_height = self._get_cell_size()
        return cell_width * (2 * column + 1) / 2, cell_height * (2 * row + 1) / 2

    def state_updated(self, state):
        self.state = state
        self.redraw()

    def on_resize(self, _event):
        self.redraw()

    def on_click(self, event):
        self.callback(*self._get_row_column_for_event(event))

    def on_motion(self, event):
        self.delete("motion")
        if not self.master.is_waiting_for_user_move:
            return
        row, column = self._get_row_column_for_event(event)
        if self.state.board[row, column] == Player.NO_PLAYER:
            self._draw_player_symbol(
                row,
                column,
                self.state.current_player,
                tag="motion",
                ratio=0.5)

    def on_leave(self, _event):
        self.delete("motion")

    def _draw_player_symbol(self, row, column, player, tag=None, ratio=1.):
        cell_width, cell_height = self._get_cell_size()
        center_x, center_y = self._get_center(row, column)
        offset = min(cell_width, cell_height) / 3
        offset *= ratio
        if player == Player.FIRST_PLAYER:
            self.create_line(
                center_x - offset,
                center_y - offset,
                center_x + offset,
                center_y + offset,
                width=3,
                fill=_get_player_color(Player.FIRST_PLAYER),
                capstyle=tk.ROUND,
                tags=tag)
            self.create_line(
                center_x - offset,
                center_y + offset,
                center_x + offset,
                center_y - offset,
                width=3,
                fill=_get_player_color(Player.FIRST_PLAYER),
                capstyle=tk.ROUND,
                tags=tag)
        elif player == Player.SECOND_PLAYER:
            self.create_oval(
                center_x - offset,
                center_y - offset,
                center_x + offset,
                center_y + offset,
                width=3,
                outline=_get_player_color(Player.SECOND_PLAYER),
                tags=tag)

    def redraw(self):
        self.delete("all")
        rows, columns = self.rules.board_size

        # draw grid
        width, height = self.winfo_width(), self.winfo_height()
        for row in range(1, rows):
            self.create_line(0, row * height / rows, width, row * height / rows, width=3)
        for column in range(1, columns):
            self.create_line(column * width / columns, 0, column * width / columns, height, width=3)

        if self.state is None:
            return

        # draw symbols
        for row in range(rows):
            for column in range(columns):
                self._draw_player_symbol(row, column, self.state.board[row, column])

        # draw winning line
        if self.state.result_extra_info is not None:
            start, end = self.state.result_extra_info
            center_start = self._get_center(*start)
            center_end = self._get_center(*end)
            self.create_line(
                center_start[0],
                center_start[1],
                center_end[0],
                center_end[1],
                width=10,
                fill=_get_player_color(self.state.board[start]),
                capstyle=tk.ROUND)
