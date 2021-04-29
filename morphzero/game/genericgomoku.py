from morphzero.game.base import Player, Rules, State, Move, GameEngine
from morphzero.common import Directions, check_all_inside_and_match
import numpy as np

class GenericGomokuRules(Rules):
    def __init__(self, board_size, goal, first_player_name, second_player_name):
        if len(board_size) != 2 or min(board_size) < 2:
            raise ValueError(f"Invalid board size: {board_size}")
        if goal < 2 or goal > min(board_size):
            raise ValueError(f"Invalid goal ({goal}) for given board size ({board_size}))")

        self.board_size = board_size
        self.goal = goal
        self.first_player_name = first_player_name
        self.second_player_name = second_player_name

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

class GenericGomokuMove(Move):
    def __init__(self, row, column):
        super().__init__((row, column))

class GenericGomokuState(State):
    def __init__(self, board, current_player, result):
        self._board = board
        self._current_player = current_player
        self._result = result

    @property
    def board(self):
        return self._board

    @property
    def current_player(self):
        return self._current_player

    @property
    def result(self):
        return self._result

    def key(self):
        return (self.board, self.current_player)

    def __str__(self):
        return str(self.board)
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

class GenericGomokuGameEngine(GameEngine):
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
                elif any(
                        check_all_inside_and_match(board, (i, j), direction, rules.goal)
                        for direction in Directions.HALF_INTERCARDINAL_DIRECTIONS):
                    return board[i, j]
        return None if has_valid_moves else Player.NO_PLAYER

    def new_game(self):
        board = np.full(self.rules.board_size, Player.NO_PLAYER)
        return GenericGomokuState(
            board,
            Player.FIRST_PLAYER,
            GenericGomokuGameEngine._get_game_result(board, self.rules))

    def is_move_playable(self, state, move):
        return not state.is_game_over and state.board[move] == Player.NO_PLAYER

    def playable_moves(self, state):
        if state.is_game_over:
            return None

        rows, columns = self.rules.board_size
        for i in range(rows):
            for j in range(columns):
                if state.board[i, j] == Player.NO_PLAYER:
                    yield (i, j)

    def play_move(self, state, move):
        if not self.is_move_playable(state, move):
            raise ValueError(f"Move ({move}) is not available for given state ({rept(state)}).")
        board = state.board.copy()
        board[move] = state.current_player
        return GenericGomokuState(
            board,
            state.current_player.other_player(),
            GenericGomokuGameEngine._get_game_result(board, self.rules))
