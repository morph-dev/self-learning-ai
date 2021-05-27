import numpy as np

from morphzero.game.base import Player


class Directions:
    RIGHT_AND_DOWN_DIRECTIONS = [(0, 1), (1, 0)]
    """Only → and ↓ directions."""

    CARDINAL_DIRECTIONS = list(zip([-1, 0, 1, 0], [0, 1, 0, -1]))
    """The 4 main directions (↑ → ↓ ←)."""

    INTERCARDINAL_DIRECTIONS = [
        (i - 1, j - 1) for (i, j) in np.ndindex((3, 3)) if (i, j) != (1, 1)
    ]
    """The 8 main directions (↖ ↑ ↗ ← → ↙ ↓ ↘)."""

    HALF_INTERCARDINAL_DIRECTIONS = list(zip([0, 1, 1, 1], [1, 1, 0, -1]))
    """Only one direction from main 8 axis (→ ↘ ↓ ↙)."""


def is_inside_matrix(index, size):
    """
    Returns whether given index is inside matrix of a given size.
    """
    return (0 <= index[0] < size[0]) and (0 <= index[1] < size[1])


def check_all_inside_and_match(matrix, start_index, delta, length):
    """
    Returns whether all length values, starting with start_index and incrementing by delta,
    """
    if not is_inside_matrix(start_index, matrix.shape):
        return False
    value = matrix[start_index]
    return all(
        is_inside_matrix(index, matrix.shape) and value == matrix[index]
        for index in generate_indices(start_index, delta, length)
    )


def generate_indices(start_index, delta, length):
    """
    Generates following indices:
    start_index, start_index + delta, ..., start_index + (length - 1) * delta
    where addition is performed element wise.
    """
    start_index = tuple(start_index)
    for i in range(length):
        yield tuple(si + i * d for si, d in zip(start_index, delta))


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
