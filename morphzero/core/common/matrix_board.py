from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Iterable, TypeVar, Generic, Mapping, Iterator, Dict, Tuple

from morphzero.core.game import Board

T = TypeVar("T")


class MatrixBoardSize(NamedTuple):
    rows: int
    columns: int

    def contains(self, coordinates: MatrixBoardCoordinates) -> bool:
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


def _to_board_coordinates_list(directions: Iterable[Tuple[int, int]]) -> Tuple[MatrixBoardCoordinates, ...]:
    return tuple(MatrixBoardCoordinates(*t) for t in directions)


RIGHT_AND_DOWN_DIRECTIONS: Tuple[MatrixBoardCoordinates, ...] = _to_board_coordinates_list(
    [(0, 1), (1, 0)])
"""Only → and ↓ directions."""

CARDINAL_DIRECTIONS: Tuple[MatrixBoardCoordinates, ...] = _to_board_coordinates_list(
    [(-1, 0), (0, 1), (1, 0), (0, -1)])
"""The 4 main directions (↑ → ↓ ←)."""

INTERCARDINAL_DIRECTIONS: Tuple[MatrixBoardCoordinates, ...] = _to_board_coordinates_list(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
"""The 8 main directions (↖ ↑ ↗ ← → ↙ ↓ ↘)."""

HALF_INTERCARDINAL_DIRECTIONS: Tuple[MatrixBoardCoordinates, ...] = _to_board_coordinates_list(
    [(0, 1), (1, 1), (1, 0), (1, -1)])
"""Only one direction from main 8 axis (→ ↘ ↓ ↙)."""


@dataclass(frozen=True)
class MatrixBoard(Board, Mapping[MatrixBoardCoordinates, T], Generic[T]):
    data: Tuple[Tuple[T, ...], ...]

    @property
    def rows(self) -> Tuple[Tuple[T, ...], ...]:
        return self.data

    @property
    def size(self) -> MatrixBoardSize:
        return MatrixBoardSize(rows=len(self.data), columns=len(self.data[0]))

    def __contains__(self, coordinates: object) -> bool:
        if not isinstance(coordinates, MatrixBoardCoordinates):
            return False
        return self.size.contains(coordinates)

    def __getitem__(self, coordinates: MatrixBoardCoordinates) -> T:
        if coordinates in self:
            return self.data[coordinates.row][coordinates.column]
        else:
            raise KeyError(f"Coordinates {coordinates} outside bounds {self.size}.")

    def __len__(self) -> int:
        size = self.size
        return size.rows * size.columns

    def __iter__(self) -> Iterator[MatrixBoardCoordinates]:
        size = self.size
        for row in range(size.rows):
            for column in range(size.columns):
                yield MatrixBoardCoordinates(row, column)

    @classmethod
    def create_empty(cls, board_size: MatrixBoardSize, default_value: T) -> MatrixBoard[T]:
        row = (default_value,) * board_size.columns
        data = (row,) * board_size.rows
        return cls(data)

    def replace(self, replacements: Dict[MatrixBoardCoordinates, T]) -> MatrixBoard[T]:
        new_data = list(list(row) for row in self.data)
        for coordinates in replacements:
            new_data[coordinates.row][coordinates.column] = replacements[coordinates]
        return type(self)(tuple(tuple(row) for row in new_data))
