from abc import ABC, abstractmethod
from collections import Iterable

from morphzero.core.game import State, Result, MoveOrMoveIndex, Rules


class Model(ABC):
    @abstractmethod
    def supports_rules(self, rules: Rules) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def play_move(self, state: State) -> MoveOrMoveIndex:
        raise NotImplementedError()


class TrainingModel(Model, ABC):
    @abstractmethod
    def train(self, result: Result, states: Iterable[State]) -> None:
        raise NotImplementedError()
