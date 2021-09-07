from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TypeVar, NamedTuple, Dict, Tuple, Iterable

from morphzero.ai.algorithms.util import pick_one_index_with_highest_value, result_for_player
from morphzero.ai.model import TrainingModel
from morphzero.core.game import State, Rules, MoveOrMoveIndex, Result

E = TypeVar('E', bound='Evaluator')
EM = TypeVar('EM', bound='EvaluatorModel')


class EvaluationResult(NamedTuple):
    """The model evaluation result for a given state.

    Attributes:
        win_rate: The estimated with rate for current player. In range: [0, 1]
        move_policy: The tuple of scores associated with each move (using move_index).
    """
    win_rate: float
    move_policy: Tuple[float, ...]

    @classmethod
    def normalize_and_create(cls, win_rate: float, move_policy: Tuple[float, ...]) -> EvaluationResult:
        """Normalizes move_policy and creates EvaluationResult.

        It's expected that all values in move_policy are non-negative.
        If at least one is positive (non-zero), all are scaled so that their sum is 1.
        """
        if any(policy < 0 for policy in move_policy):
            raise ValueError("All move policies should be non-negative.")
        policy_sum = sum(move_policy)
        return EvaluationResult(
            win_rate=win_rate,
            move_policy=tuple(policy / policy_sum for policy in move_policy),
        )


class Evaluator(ABC):
    """Base class for evaluating game state."""

    @abstractmethod
    def supports_rules(self, rules: Rules) -> bool:
        """Whether this model supports given rules."""
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, state: State) -> EvaluationResult:
        """Evaluates the state."""
        raise NotImplementedError()

    @abstractmethod
    def train(self, learning_data: Dict[State, EvaluationResult]) -> None:
        """Allows Evaluator to improve it's evaluation.

        Args:
            learning_data: The desired State -> EvaluationResult based on the played game.
        """
        raise NotImplementedError()


class EvaluatorModel(TrainingModel, ABC):
    """Model that is based on the Evaluator."""
    evaluator: Evaluator
    move_picker: EvaluatorModel.MovePicker

    def __init__(self, evaluator: Evaluator, move_picker: EvaluatorModel.MovePicker):
        self.evaluator = evaluator
        self.move_picker = move_picker

    def supports_rules(self, rules: Rules) -> bool:
        return self.evaluator.supports_rules(rules)

    def play_move(self, state: State) -> MoveOrMoveIndex:
        evaluation_result = self.evaluator.evaluate(state)

        return self.move_picker.pick_move(evaluation_result)

    def train_from_game(self, result: Result, states: Iterable[State]) -> None:
        learning_data = {
            state: EvaluationResult.normalize_and_create(
                result_for_player(state.current_player, result),
                tuple()
            )
            for state in states
        }
        return self.evaluator.train(learning_data)

    @classmethod
    def create_with_best_move_picker(cls, evaluator: Evaluator) -> EvaluatorModel:
        """Creates the EvaluatorModel that picks the move based on highest move_policy."""
        return EvaluatorModel(evaluator, BestMovePicker())

    @classmethod
    def create_with_probability_move_picker(cls, evaluator: Evaluator) -> EvaluatorModel:
        """Creates the EvaluatorModel that picks the move according to probability (move_policy)."""
        return EvaluatorModel(evaluator, ProbabilityMovePicker())

    class MovePicker(ABC):
        """Picks a move from evaluation_result."""

        @abstractmethod
        def pick_move(self, evaluation_result: EvaluationResult) -> int:
            """Returns selected move."""
            raise NotImplementedError()


class BestMovePicker(EvaluatorModel.MovePicker):
    """Picks move with the highest move_policy."""

    def pick_move(self, evaluation_result: EvaluationResult) -> int:
        move_policy = evaluation_result.move_policy
        move_index = pick_one_index_with_highest_value(move_policy)
        return move_index


class ProbabilityMovePicker(EvaluatorModel.MovePicker):
    """Picks the move according to probability.

    Each move has a probability equal to move_policy.
    """

    def pick_move(self, evaluation_result: EvaluationResult) -> int:
        move_policy = evaluation_result.move_policy
        return random.choices(range(len(move_policy)), weights=move_policy)[0]
