import numpy as np

from game.common import MatrixBoardConnectInARowRules, MatrixState, ConnectInARowResult
from morphzero.common import BoardCoordinates
from morphzero.game.base import Player, GameEngine


class GenericGomokuRules(MatrixBoardConnectInARowRules):
    def create_game_engine(self):
        return GenericGomokuGameEngine(self)

    @classmethod
    def create_tic_tac_toe_rules(cls):
        return cls(board_size=(3, 3), goal=3)

    @classmethod
    def create_gomoku_rules(cls):
        return cls(board_size=(15, 15), goal=5)


GenericGomokuMove = BoardCoordinates


class GenericGomokuState(MatrixState):
    pass


class GenericGomokuResult(ConnectInARowResult):
    pass


class GenericGomokuGameEngine(GameEngine):
    def __init__(self, rules):
        super().__init__(rules)

    def new_game(self):
        board = np.full(self.rules.board_size, Player.NO_PLAYER)
        return GenericGomokuState(
            board=board,
            current_player=Player.FIRST_PLAYER,
            result=None)

    def is_move_playable(self, state, move):
        return not state.is_game_over and state.board[move] == Player.NO_PLAYER

    def playable_moves(self, state):
        if state.is_game_over:
            return None

        rows, columns = self.rules.board_size
        for row in range(rows):
            for column in range(columns):
                if state.board[row, column] == Player.NO_PLAYER:
                    yield GenericGomokuMove(row=row, column=column)

    def play_move(self, state, move):
        if not self.is_move_playable(state, move):
            raise ValueError(f"Move ({move}) is not available for given state ({state}).")
        board = state.board.copy()
        board[move] = state.current_player
        return GenericGomokuState(
            board=board,
            current_player=state.current_player.other_player,
            result=ConnectInARowResult.create_from_board_and_last_move(
                self.rules, board, move))
