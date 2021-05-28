import numpy as np

from morphzero.common import Directions, check_all_inside_and_match, print_board
from morphzero.game.base import Player, Rules, State, Move, GameEngine


class ConnectFourRules(Rules):
    def __init__(self, board_size, goal):
        if len(board_size) != 2 or min(board_size) < 2:
            raise ValueError(f"Invalid board size: {board_size}")
        if goal < 2 or goal > min(board_size):
            raise ValueError(f"Invalid goal ({goal}) for given board size ({board_size}))")

        self.board_size = board_size
        self.goal = goal

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class ConnectFourMove(Move):
    def __init__(self, column):
        super().__init__(column)


class ConnectFourState(State):
    def __init__(self, board, current_player, result, result_extra_info):
        self._board = board
        self._current_player = current_player
        self._result = result
        self._result_extra_info = result_extra_info

    @property
    def board(self):
        return self._board

    @property
    def current_player(self):
        return self._current_player

    @property
    def result(self):
        return self._result

    @property
    def result_extra_info(self):
        """
        Returns two tuples (row, column), representing the end location of the winning sequence.
        """
        return self._result_extra_info

    def key(self):
        return self.current_player, tuple(self.board.flatten())

    def __str__(self):
        return print_board(self.board)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)


class ConnectFourGameEngine(GameEngine):
    def __init__(self, rules):
        super().__init__(rules)

    @staticmethod
    def _get_game_result(board, rules):
        rows, columns = board.shape
        has_valid_moves = False
        for i in range(rows):
            for j in range(columns):
                if board[i, j] == Player.NO_PLAYER:
                    has_valid_moves = True
                else:
                    for direction in Directions.HALF_INTERCARDINAL_DIRECTIONS:
                        if check_all_inside_and_match(board, (i, j), direction, rules.goal):
                            start = i, j
                            end = (
                                i + direction[0] * (rules.goal - 1),
                                j + direction[1] * (rules.goal - 1)
                            )
                            return board[i, j], (start, end)
        return (None, None) if has_valid_moves else (Player.NO_PLAYER, None)

    def new_game(self):
        board = np.full(self.rules.board_size, Player.NO_PLAYER)
        result, result_extra_info = ConnectFourGameEngine._get_game_result(board, self.rules)
        return ConnectFourState(
            board,
            Player.FIRST_PLAYER,
            result,
            result_extra_info)

    def is_move_playable(self, state, move):
        return not state.is_game_over and state.board[0, move.key] == Player.NO_PLAYER

    def playable_moves(self, state):
        if state.is_game_over:
            return None

        _, columns = self.rules.board_size
        for column in range(columns):
            if state.board[0, column] == Player.NO_PLAYER:
                yield ConnectFourMove(column)

    def play_move(self, state, move):
        if not self.is_move_playable(state, move):
            raise ValueError(f"Move ({move}) is not available for given state ({repr(state)}).")
        board = state.board.copy()
        board[0, move.key] = state.current_player
        result, result_extra_info = ConnectFourGameEngine._get_game_result(board, self.rules)
        return ConnectFourState(
            board,
            state.current_player.other_player(),
            result,
            result_extra_info)
