from abc import ABC, abstractmethod
from collections import Iterable

from morphzero.core.game import State, Result, MoveOrMoveIndex, Rules


class Model(ABC):
    """Base class for playing the game."""

    @abstractmethod
    def supports_rules(self, rules: Rules) -> bool:
        """Whether given rules are supported."""
        raise NotImplementedError()

    @abstractmethod
    def play_move(self, state: State) -> MoveOrMoveIndex:
        """Returns move played from a given state."""
        raise NotImplementedError()


class TrainingModel(Model, ABC):
    """Base class for playing the game and learning based on played games."""

    @abstractmethod
    def train(self, result: Result, states: Iterable[State]) -> None:
        raise NotImplementedError()
