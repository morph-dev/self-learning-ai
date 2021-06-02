from collections import namedtuple

import numpy as np

from morphzero.common import BoardCoordinates
from morphzero.game.base import Player, GameEngine
from morphzero.game.common import ConnectInARowResult, MatrixState, MatrixBoardConnectInARowRules


class ConnectFourRules(MatrixBoardConnectInARowRules):
    def create_game_engine(self):
        return ConnectFourGameEngine(self)

    @classmethod
    def create_default_rules(cls):
        return cls(board_size=(6, 7), goal=4)


ConnectFourMove = namedtuple("ConnectFourMove", ["column"])


class ConnectFourState(MatrixState):
    pass


class ConnectFourResult(ConnectInARowResult):
    pass


class ConnectFourGameEngine(GameEngine):
    def __init__(self, rules):
        super().__init__(rules)

    def new_game(self):
        board = np.full(self.rules.board_size, Player.NO_PLAYER)
        return ConnectFourState(
            board=board,
            current_player=Player.FIRST_PLAYER,
            result=None)

    def is_move_playable(self, state, move):
        return not state.is_game_over and state.board[0, move.column] == Player.NO_PLAYER

    def playable_moves(self, state):
        if state.is_game_over:
            return None

        _, columns = self.rules.board_size
        for column in range(columns):
            if state.board[0, column] == Player.NO_PLAYER:
                yield ConnectFourMove(column)

    def play_move(self, state, move):
        if not self.is_move_playable(state, move):
            raise ValueError(f"Move ({move}) is not available for given state ({state}).")
        board = state.board.copy()
        rows, _ = self.rules.board_size
        row = rows - 1
        while board[row, move.column] != Player.NO_PLAYER:
            row -= 1
        coordinates = BoardCoordinates(row, move.column)
        board[coordinates] = state.current_player
        return ConnectFourState(
            board=board,
            current_player=state.current_player.other_player,
            result=ConnectInARowResult.create_from_board_and_last_move(
                self.rules, board, coordinates))
