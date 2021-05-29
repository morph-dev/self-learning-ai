from collections import namedtuple

from morphzero.game.base import Player


class BoardCoordinates(namedtuple("BoardCoordinates", ["row", "column"])):
    __slots__ = ()

    def __add__(self, other):
        return BoardCoordinates(row=self.row + other.row,
                                column=self.column + other.column)

    def __sub__(self, other):
        return BoardCoordinates(row=self.row - other.row,
                                column=self.column - other.column)

    def __mul__(self, mul):
        return BoardCoordinates(row=self.row * mul,
                                column=self.column * mul)

    def __neg__(self):
        return BoardCoordinates(row=-self.row,
                                column=-self.column)


def _to_board_coordinates_list(directions):
    return [BoardCoordinates(*t) for t in directions]


class Directions:
    RIGHT_AND_DOWN = _to_board_coordinates_list([
        (0, 1), (1, 0)])
    """Only → and ↓ directions."""

    CARDINAL = _to_board_coordinates_list([
        (-1, 0), (0, 1), (1, 0), (0, -1)])
    """The 4 main directions (↑ → ↓ ←)."""

    INTERCARDINAL = _to_board_coordinates_list([
        (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    """The 8 main directions (↖ ↑ ↗ ← → ↙ ↓ ↘)."""

    HALF_INTERCARDINAL = _to_board_coordinates_list([
        (0, 1), (1, 1), (1, 0), (1, -1)])
    """Only one direction from main 8 axis (→ ↘ ↓ ↙)."""

    @classmethod
    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("Can't create instance of the Directions class.")


def is_inside_matrix(index, size):
    """
    Returns whether given index is inside matrix of a given size.
    """
    return (0 <= index[0] < size[0]) and (0 <= index[1] < size[1])


def check_all_inside_and_match(board, start_coordinates, delta, length):
    """
    Returns whether all coordinates, starting with start_index and incrementing
    by delta length times, have the same value in the board.
    """
    if not is_inside_matrix(start_coordinates, board.shape):
        return False
    value = board[start_coordinates]
    return all(
        is_inside_matrix(coordinates, board.shape) and value == board[coordinates]
        for coordinates in generate_coordinates(start_coordinates, delta, length)
    )


def generate_coordinates(start_coordinates, delta, length):
    """
    Generates following coordinates:
    start_coordinates, start_coordinates + delta, ..., start_coordinates + (length - 1) * delta
    """
    coordinates = start_coordinates
    for i in range(length):
        yield coordinates
        coordinates += delta


def print_board(board, no_player_symbol=" ", first_player_symbol="X", second_player_symbol="O"):
    """
    Prints human readable board with '|' and '-' as separators between cells and fills the rest
    using appropriate symbols.
    """
    row_separator = "-" * (board.shape[1] * 4 - 1)
    player_map = {
        Player.NO_PLAYER: no_player_symbol,
        Player.FIRST_PLAYER: first_player_symbol,
        Player.SECOND_PLAYER: second_player_symbol,
    }

    def row_converter(row):
        return '|'.join(f" {player_map[v]} " for v in row)

    return ("\n" + row_separator + "\n").join(row_converter(row) for row in board)
