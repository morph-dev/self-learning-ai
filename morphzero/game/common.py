from collections import namedtuple

from common import print_board, BoardCoordinates, Directions, check_all_inside_and_match, is_inside_matrix
from game.base import Rules, State, Player


class MatrixBoardConnectInARowRules(Rules,
                                    namedtuple(
                                        "MatrixBoardConnectInARowRules",
                                        ["board_size", "goal"])):
    """
    Base Rules class for games that require matrix board and
    whose goal is to connect certain amount of pieces in a row.
    """
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        if len(instance.board_size) != 2 or min(instance.board_size) < 2:
            raise ValueError(f"Invalid board size: {instance.board_size}")
        if instance.goal < 2 or instance.goal > min(instance.board_size):
            raise ValueError(f"Invalid goal ({instance.goal}) for given board size: {instance.board_size}")
        return instance

    def create_game_engine(self):
        raise NotImplementedError()


class MatrixState(State):
    def __init__(self, board, current_player, result):
        super().__init__(current_player, result)
        self._board = board

    @property
    def board(self):
        return self._board

    def key(self):
        return self.current_player, tuple(self.board.flatten())

    def __format__(self, format_spec):
        # normal
        if format_spec == "":
            return f"{self.board}"
        # full / repr
        if format_spec == "r":
            return f"{self.__class__}(board:{self.board}; current_player:{self.current_player}; result:{self.result})"
        # terminal
        if format_spec == "t":
            return print_board(self.board)
        raise ValueError(f"Unsupported format ({format_spec}) passed to {self.__class__}.")

    def __str__(self):
        return self.__format__("")

    def __repr__(self):
        return self.__format__("r")


class ConnectInARowResult(namedtuple("ConnectInARowResult",
                                     ["winner", "winning_line_start", "winning_line_end"],
                                     defaults=[None, None])):
    """
    Result type for games where the goal is to connect several pieces in a row.
    """
    __slots__ = ()

    @classmethod
    def create_from_board(cls, rules, board):
        rows, columns = board.shape
        has_valid_moves = False
        for row in range(rows):
            for column in range(columns):
                coordinates = BoardCoordinates(row, column)
                if board[coordinates] == Player.NO_PLAYER:
                    has_valid_moves = True
                else:
                    for direction in Directions.HALF_INTERCARDINAL:
                        if check_all_inside_and_match(board, coordinates, direction, rules.goal):
                            end = coordinates + (direction * (rules.goal - 1))
                            return cls(winner=board[coordinates],
                                       winning_line_start=coordinates,
                                       winning_line_end=end)
        if has_valid_moves:
            return None
        else:
            return cls(Player.NO_PLAYER)

    @classmethod
    def create_from_board_and_last_move(cls, rules, board, last_move_coordinates):
        last_move_coordinates = BoardCoordinates(*last_move_coordinates)
        if board[last_move_coordinates] == Player.NO_PLAYER:
            raise ValueError("Board can't be empty at the coordinates of the last move.")

        for direction in Directions.HALF_INTERCARDINAL:
            start = last_move_coordinates
            while True:
                new_start = start - direction
                if not is_inside_matrix(new_start, board.shape) or board[new_start] != board[last_move_coordinates]:
                    break
                start = new_start
            if check_all_inside_and_match(board, start, direction, rules.goal):
                end = start + (direction * (rules.goal - 1))
                return cls(winner=board[last_move_coordinates],
                           winning_line_start=start,
                           winning_line_end=end)
        if (board == Player.NO_PLAYER).any():
            return None
        else:
            return cls(winner=Player.NO_PLAYER)
