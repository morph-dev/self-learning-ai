from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, DefaultDict, Union, Iterator

import numpy as np

from morphzero.core.common import matrix_board
from morphzero.core.common.matrix_board import MatrixBoardCoordinates, MatrixBoardSize
from morphzero.core.game import Rules, State, Player, Result, Move, Engine, FIRST_OR_SECOND_PLAYER


@dataclass(frozen=True)
class ConnectOnMatrixBoardResult(Result):
    """Result type for games where the goal is to connect pieces on matrix board.

    Attributes:
        winning_coordinates: The coordinates of the pieces that are making required connection.
    """
    winning_coordinates: Optional[tuple[MatrixBoardCoordinates, ...]] = None

    @classmethod
    def create_resignation(cls, winner: Player) -> ConnectOnMatrixBoardResult:
        """Creates the resignation result."""
        return cls(winner=winner, resignation=True)

    @classmethod
    def create_from_board(
            cls, rules: ConnectOnMatrixBoardRules, board: np.ndarray
    ) -> Optional[ConnectOnMatrixBoardResult]:
        """Checks the board and creates the result object accordingly.

        Args:
            rules: Rules used to determine the result.
            board: The state of the board.

        Returns:
            None if game is not over, otherwise result.
        """
        board_size = rules.board_size
        if board.size != board_size:
            raise ValueError(f"The rules.board_size ({board_size}) and board.size ({board.size}) don't match.")

        # → ↘ ↓ ↙
        directions = matrix_board.HALF_INTERCARDINAL_DIRECTIONS
        # count[player][coordinate][direction_index] ->
        # how many consecutive cells ending with 'coordinate' in 'direction_index' have 'player' value.
        count = DefaultDict[Player, np.ndarray](lambda: np.zeros(
            shape=(board_size.rows, board_size.columns, len(directions)),
            dtype=int))
        has_valid_moves = False
        for row in range(board_size.rows):
            for column in range(board_size.columns):
                coordinate = MatrixBoardCoordinates(row, column)
                player = board[coordinate]
                if player == Player.NO_PLAYER:
                    has_valid_moves = True
                    continue

                count[player][coordinate].fill(1)
                for direction_index, direction in enumerate(directions):
                    if (coordinate - direction) in board_size:
                        count[player][coordinate][direction] += count[player][coordinate - direction][direction]
                    if count[player][coordinate][direction] >= rules.goal:
                        winning_coordinates = tuple(
                            coordinate - (direction * i) for i in reversed(range(rules.goal))
                        )
                        return cls(winner=player,
                                   winning_coordinates=winning_coordinates)

        if has_valid_moves:
            return None
        else:
            return cls(winner=Player.NO_PLAYER)

    @classmethod
    def create_from_board_and_last_move(
            cls, rules: ConnectOnMatrixBoardRules, board: np.ndarray, last_move_coordinates: MatrixBoardCoordinates
    ) -> Optional[ConnectOnMatrixBoardResult]:
        """Creates the result based on last move.

        We don't check any winning combination that doesn't involve last player move and assume that game wasn't over
        before that move was played.

        Args:
            rules: The rules used to determine the result.
            board: The state of the board after move was played.
            last_move_coordinates: The coordinates of the last move.

        Returns:
            None if game is not over, otherwise result.
        """
        board_size = rules.board_size
        if board.shape != board_size:
            raise ValueError(f"The rules.board_size ({board_size}) and board.size ({board.size}) don't match.")

        player = board[last_move_coordinates]
        if player == Player.NO_PLAYER:
            raise ValueError("Board can't be empty at the coordinates of the last move.")

        # → ↘ ↓ ↙
        directions = matrix_board.HALF_INTERCARDINAL_DIRECTIONS
        for direction in directions:

            def move_while_it_matches(d: MatrixBoardCoordinates) -> tuple[MatrixBoardCoordinates, int]:
                """Returns last coordinates that matched and how many times in moved."""
                result = last_move_coordinates
                move_count = 0
                while True:
                    new_result = result + d
                    if new_result not in board_size or board[new_result] != player:
                        break
                    result = new_result
                    move_count += 1
                return result, move_count

            start, move_count_start = move_while_it_matches(-direction)
            _, move_count_end = move_while_it_matches(direction)
            connected = 1 + move_count_start + move_count_end
            if connected >= rules.goal:
                winning_coordinates = tuple(
                    start + (direction * i) for i in range(connected)
                )
                return cls(winner=player,
                           winning_coordinates=winning_coordinates)

        if (board == Player.NO_PLAYER).any():
            return None
        else:
            return cls(winner=Player.NO_PLAYER)


@dataclass(frozen=True)
class ConnectOnMatrixBoardState(State):
    """The base class for game states for games that are played on a matrix board.

    Attributes:
        board: The status of the board.
    """
    result: Optional[ConnectOnMatrixBoardResult]
    board: np.ndarray

    def __post_init__(self) -> None:
        self.board.setflags(write=False)

    def _key(self) -> tuple[FIRST_OR_SECOND_PLAYER, Optional[ConnectOnMatrixBoardResult], tuple[Player, ...]]:
        return self.current_player, self.result, tuple(self.board.flatten())

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConnectOnMatrixBoardState):
            return False
        return self._key() == other._key()


@dataclass(frozen=True)
class ConnectOnMatrixBoardMove(Move):
    """The base class for moves for games that are played on a matrix board.

    Attributes:
        coordinates: The coordinates of the played move. It is None if it is resignation.
    """
    coordinates: Optional[MatrixBoardCoordinates] = field(compare=False)

    def __post_init__(self) -> None:
        assert self.resign ^ (self.coordinates is not None), "Either it's resignation move or it has coordinates"


MoveOrMoveIndex = Union[ConnectOnMatrixBoardMove, int]


@dataclass(frozen=True)
class ConnectOnMatrixBoardRules(Rules, ABC):
    """Base class for rules for games that are played on a matrix board and goal of players is to connect pieces.

    Attributes:
        board_size: The board size.
        goal: How many pieces are required to be connected in order to win the game.
    """

    board_size: MatrixBoardSize
    goal: int

    def __post_init__(self) -> None:
        if min(self.board_size) < 2:
            raise ValueError(f"Board needs to be at least 2x2. Actual: {self.board_size}.")
        if not (2 <= self.goal <= max(self.board_size)):
            raise ValueError(f"Goal ({self.goal}) for connection has to be at least 2 "
                             f"and at most bigger size of the board ({self.board_size}).")

    def number_of_possible_moves(self) -> int:
        raise NotImplementedError()

    def create_engine(self) -> ConnectOnMatrixBoardEngine:
        raise NotImplementedError()


class ConnectOnMatrixBoardEngine(Engine, ABC):
    """Base class for engine for games that are played on a matrix board where played adds new pieces.

    This base class mostly handles conversion between move_index and move coordinates.
    """
    rules: ConnectOnMatrixBoardRules

    @staticmethod
    def number_of_possible_moves(board_size: MatrixBoardSize) -> int:
        """Returns number of possible moves for a given board size.

        It's equal to number of possible cells + 1 for resignation.
        """
        return board_size.rows * board_size.columns + 1

    def get_move_index_for_resign(self) -> int:
        """Returns move index for resign move."""
        board_size = self.rules.board_size
        return board_size.rows * board_size.columns

    def get_move_index_for_coordinates(self, coordinates: MatrixBoardCoordinates) -> int:
        """Returns move index for given coordinates.

        Raises:
            ValueError: When coordinates are outside rules.board_size
        """
        if coordinates not in self.rules.board_size:
            raise ValueError(f"Invalid coordinates {coordinates}.")
        return coordinates.row * self.rules.board_size.columns + coordinates.column

    def get_coordinates_for_move_index(self, move_index: int) -> Optional[MatrixBoardCoordinates]:
        """Returns coordinates for valid move_index.

        Returns:
            Coordinates of the move, or None if move is resignation.

        Raises:
            ValueError: When invalid move_index is passed.
        """
        board_size = self.rules.board_size
        if not (0 <= move_index < self.number_of_possible_moves(board_size)):
            raise ValueError(f"Invalid move index: {move_index}")
        if move_index == self.get_move_index_for_resign():
            return None
        row = move_index // board_size.columns
        column = move_index % board_size.columns
        return MatrixBoardCoordinates(row, column)

    def validate_move(self, move: ConnectOnMatrixBoardMove) -> None:
        if self.get_coordinates_for_move_index(move.move_index) != move.coordinates:
            raise ValueError(f"Move coordinates ({move.coordinates}) don't match move_index ({move.move_index})")
        if move.resign != (self.get_move_index_for_resign() == move.move_index):
            raise ValueError(f"Resign status is not correct.")

    def new_game(self) -> ConnectOnMatrixBoardState:
        raise NotImplementedError()

    def create_move_from_move_index(self, move_index: int) -> ConnectOnMatrixBoardMove:
        raise NotImplementedError()

    def playable_moves_bitmap(self, state: ConnectOnMatrixBoardState) -> tuple[bool, ...]:  # type: ignore[override]
        raise NotImplementedError()

    def playable_moves(  # type: ignore[override]
            self, state: ConnectOnMatrixBoardState) -> Iterator[ConnectOnMatrixBoardMove]:
        raise NotImplementedError()

    def is_move_playable(  # type: ignore[override]
            self, state: ConnectOnMatrixBoardState, move: MoveOrMoveIndex) -> bool:
        raise NotImplementedError()

    def play_move(  # type: ignore[override]
            self, state: ConnectOnMatrixBoardState, move: MoveOrMoveIndex) -> ConnectOnMatrixBoardState:
        raise NotImplementedError()
