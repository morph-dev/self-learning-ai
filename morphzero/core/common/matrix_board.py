from __future__ import annotations

from typing import NamedTuple, Iterable


class MatrixBoardSize(NamedTuple):
    rows: int
    columns: int

    def __contains__(self, coordinates: object) -> bool:
        if not isinstance(coordinates, MatrixBoardCoordinates):
            return False
        return (0 <= coordinates.row < self.rows) and (0 <= coordinates.column < self.columns)


class MatrixBoardCoordinates(NamedTuple):
    row: int
    column: int

    def __add__(self, other: object) -> MatrixBoardCoordinates:
        if not isinstance(other, MatrixBoardCoordinates):
            raise TypeError(f"Invalid argument type: {other}")
        return MatrixBoardCoordinates(row=self.row + other.row,
                                      column=self.column + other.column)

    def __sub__(self, other: object) -> MatrixBoardCoordinates:
        if not isinstance(other, MatrixBoardCoordinates):
            raise TypeError(f"Invalid argument type: {other}")
        return MatrixBoardCoordinates(row=self.row - other.row,
                                      column=self.column - other.column)

    def __mul__(self, mul: object) -> MatrixBoardCoordinates:
        if not isinstance(mul, int):
            raise TypeError(f"Invalid argument type: {mul}")
        return MatrixBoardCoordinates(row=self.row * mul,
                                      column=self.column * mul)

    def __neg__(self) -> MatrixBoardCoordinates:
        return MatrixBoardCoordinates(row=-self.row,
                                      column=-self.column)


def _to_board_coordinates_list(directions: Iterable[tuple[int, int]]) -> tuple[MatrixBoardCoordinates, ...]:
    return tuple(MatrixBoardCoordinates(*t) for t in directions)


RIGHT_AND_DOWN_DIRECTIONS: tuple[MatrixBoardCoordinates, ...] = _to_board_coordinates_list(
    [(0, 1), (1, 0)])
"""Only → and ↓ directions."""

CARDINAL_DIRECTIONS: tuple[MatrixBoardCoordinates, ...] = _to_board_coordinates_list(
    [(-1, 0), (0, 1), (1, 0), (0, -1)])
"""The 4 main directions (↑ → ↓ ←)."""

INTERCARDINAL_DIRECTIONS: tuple[MatrixBoardCoordinates, ...] = _to_board_coordinates_list(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
"""The 8 main directions (↖ ↑ ↗ ← → ↙ ↓ ↘)."""

HALF_INTERCARDINAL_DIRECTIONS: tuple[MatrixBoardCoordinates, ...] = _to_board_coordinates_list(
    [(0, 1), (1, 1), (1, 0), (1, -1)])
"""Only one direction from main 8 axis (→ ↘ ↓ ↙)."""
