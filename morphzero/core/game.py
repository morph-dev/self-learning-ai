from __future__ import annotations

from abc import abstractmethod, ABC
from collections import Iterator
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import Optional, Union, Literal

import numpy as np


@unique
class Player(IntEnum):
    """Constants for different players."""
    NO_PLAYER = 0
    FIRST_PLAYER = 1
    SECOND_PLAYER = -1

    @property
    def other_player(self) -> FIRST_OR_SECOND_PLAYER:
        if self == Player.FIRST_PLAYER:
            return Player.SECOND_PLAYER
        elif self == Player.SECOND_PLAYER:
            return Player.FIRST_PLAYER
        else:
            raise ValueError(f"The {self} doesn't have other player.")


FIRST_OR_SECOND_PLAYER = Literal[Player.FIRST_PLAYER, Player.SECOND_PLAYER]


@dataclass(frozen=True)
class Result:
    """Contains information regarding the result of the game.

    Different games should subclass this one and add appropriate fields.

    This class should be Immutable.

    Attributes:
        winner: Can be Player.NO_PLAYER if it is a draw.
        resignation: Whether game ended by resignation.
    """
    winner: Player
    resignation: bool = False

    @property
    def is_draw(self) -> bool:
        return self.winner == Player.NO_PLAYER


class Board:
    """The board state of the game."""
    pass


@dataclass(frozen=True)
class State:
    """Uniquely represents the state of the game.

    This class should be Immutable.

    Attributes:
        current_player: The Player that is supposed to make next action (If game is over, it should be opposite from the
            player who made the last move).
        result: The Result of the game if the game is over, otherwise None.
        board: The status of the board.
    """
    current_player: FIRST_OR_SECOND_PLAYER
    result: Optional[Result]
    board: Board

    @property
    def is_game_over(self) -> bool:
        """
        Returns whether game is over.
        """
        return self.result is not None

    def to_training_data(self) -> np.array:
        """Returns np.array that is used to create tf.Tensor for training."""
        raise NotImplementedError()

@dataclass(frozen=True)
class Move:
    """Uniquely represents the move played in the game.

    Only move_index should be used for comparison and hashing. Other fields (added by subclasses) should be there for
    easier understanding what the move is doing.

    Move should be dependant on the context (current State of the game). This means that Move shouldn't contain
    information regarding which player is doing the move, what pieces is being played, whether there are captures, etc.
    Ideally, it should contain only coordinates of the move.

    This class should be Immutable.

    Attributes:
        move_index: The move_index that indicates to Engine the move to be played.
        resign: Whether Move represents the resignation.
    """
    move_index: int = field(compare=True)
    resign: bool = field(compare=False)


MoveOrMoveIndex = Union[Move, int]


class Rules(ABC):
    """Properties of the game (board size, winning_condition etc)."""

    @abstractmethod
    def number_of_possible_moves(self) -> int:
        """The total number of different moves game allows.

        At least one move has to be resignation.
        Engine provides mapping from move_index to actual move and move itself has move_index.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_engine(self) -> Engine:
        """Creates game engine."""
        raise NotImplementedError()


class Engine(ABC):
    """The Game Engine contains the logic of the game.

    Attributes:
        rules: Rules of the game.
    """
    rules: Rules

    def __init__(self, rules: Rules) -> None:
        self.rules = rules

    @abstractmethod
    def new_game(self) -> State:
        """Returns the State that corresponds to the state of the new game according to the rules."""
        raise NotImplementedError()

    @abstractmethod
    def get_move_index_for_resign(self) -> int:
        """Returns move index for a move that represents the resign."""
        raise NotImplementedError()

    @abstractmethod
    def create_move_from_move_index(self, move_index: int) -> Move:
        """Creates and returns Move that is corresponding to the move index.

        Args:
            move_index: The index of the move we are interested in.

        Returns:
            The Move corresponding to the given move_index.

        Raises:
             ValueError: If move_index is not in the correct range, which is [0, Rules.number_of_possible_moves).
        """
        raise NotImplementedError()

    @abstractmethod
    def playable_moves_bitmap(self, state: State) -> tuple[bool, ...]:
        """Indicates which moves are playable from the given state.

        If state represents the "Game Over" state, all values will be False.

        Returns:
            Boolean tuple of length Rules.number_of_possible_moves that indicates which moves are playable.
        """
        raise NotImplementedError()

    @abstractmethod
    def playable_moves(self, state: State) -> Iterator[Move]:
        """Returns iterator of playable moves for a given state.

        If game is not over, it will contain at east one Move. If game is over it will be empty.
        It's not safe to assume that the result can be iterated multiple times.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_move_playable(self, state: State, move: MoveOrMoveIndex) -> bool:
        """Returns whether move is playable from a given state.

        Args:
            state: The game state we are interested in.
            move: Can be ether move_index or actual move.

        Raises:
            If move is not valid for given rules.
        """
        raise NotImplementedError()

    @abstractmethod
    def play_move(self, state: State, move: MoveOrMoveIndex) -> State:
        """Returns the state of the game that happens after playing given move from the given state.

        Args:
            state: The game state before the move.
            move: Can be ether move_index or actual move.

        Returns:
            The game state after the move.

        Raises:
            ValueError: If move is not valid or not playable from the given state.
        """
        raise NotImplementedError()
