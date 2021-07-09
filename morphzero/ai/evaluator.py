from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Iterable
from typing import TypeVar, NamedTuple

from morphzero.ai.algorithms.util import pick_one_of_highest
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
    move_policy: tuple[float, ...]

    def normalized_move_policy(self, playable_moves: tuple[bool, ...]) -> tuple[float, ...]:
        """Normalizes move_policy taking into consideration playable moves.

        Only playable moves will be non-zero and others will be scaled so that their sum is one.
        That means that values can be interpreted as probabilities and can be used for picking next move to play.
        """
        if len(self.move_policy) == len(playable_moves):
            raise ValueError(f"Unexpected length of playable_moves ({len(playable_moves)})")
        if not any(playable_moves):
            raise ValueError("At least one playable move is expected.")

        playable_move_policy = tuple(
            policy if playable and policy >= 0 else 0
            for playable, policy in zip(playable_moves, self.move_policy)
        )
        policy_sum = sum(playable_move_policy)
        if policy_sum == 0:
            # all playable moves have equal probability
            playable_move_policy = tuple(1 if playable else 0 for playable in playable_moves)
            policy_sum = sum(playable_move_policy)
        return tuple(
            policy / policy_sum
            for policy in playable_move_policy
        )


class Evaluator(ABC):

    @abstractmethod
    def supports_rules(self, rules: Rules) -> bool:
        """Whether this model supports given rules."""
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, state: State) -> EvaluationResult:
        """Evaluates the state."""
        raise NotImplementedError()

    @abstractmethod
    def train(self, result: Result, states: Iterable[State]) -> None:
        raise NotImplementedError()


class EvaluatorModel(TrainingModel, ABC):
    evaluator: Evaluator
    move_picker: EvaluatorModel.MovePicker

    def __init__(self, evaluator: Evaluator, move_picker: EvaluatorModel.MovePicker):
        self.evaluator = evaluator
        self.move_picker = move_picker

    def supports_rules(self, rules: Rules) -> bool:
        return self.evaluator.supports_rules(rules)

    def play_move(self, state: State) -> MoveOrMoveIndex:
        evaluation_result = self.evaluator.evaluate(state)

        # # TODO: remove
        # from morphzero.common import board_to_string
        # print(board_to_string(state.board))
        # print(evaluation_result.win_rate)
        # for i, p in enumerate(evaluation_result.move_policy):
        #     print('\t', i, p)
        # print()

        return self.move_picker.pick_move(evaluation_result)

    def train(self, result: Result, states: Iterable[State]) -> None:
        return self.evaluator.train(result, states)

    @classmethod
    def create_with_best_move_picker(cls, evaluator: Evaluator) -> EvaluatorModel:
        return EvaluatorModel(evaluator, BestMovePicker())

    @classmethod
    def create_with_probability_move_picker(cls, evaluator: Evaluator) -> EvaluatorModel:
        return EvaluatorModel(evaluator, ProbabilityMovePicker())

    class MovePicker(ABC):
        @abstractmethod
        def pick_move(self, evaluation_result: EvaluationResult) -> int:
            raise NotImplementedError()


class BestMovePicker(EvaluatorModel.MovePicker):
    def pick_move(self, evaluation_result: EvaluationResult) -> int:
        move_policy = evaluation_result.move_policy
        move_index = pick_one_of_highest(
            range(len(move_policy)),
            key=lambda i: move_policy[i])
        return move_index


class ProbabilityMovePicker(EvaluatorModel.MovePicker):
    def pick_move(self, evaluation_result: EvaluationResult) -> int:
        move_policy = evaluation_result.move_policy
        return random.choices(range(len(move_policy)), weights=move_policy)[0]
