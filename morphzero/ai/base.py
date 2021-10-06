from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Iterable, Optional

from morphzero.core.game import State, Rules, Result, MoveOrMoveIndex


@dataclass(frozen=True)
class EvaluationResult:
    """The model evaluation result for a given state.

    Attributes:
        win_rate: The estimated with rate for current player. In range: [0, 1]
        move_policy: The tuple of scores associated with each move (using move_index).
    """
    win_rate: float
    move_policy: Tuple[float, ...]

    def __post_init__(self) -> None:
        assert 0 <= self.win_rate <= 1, "Win rate has to be between 0 and 1 (inclusive)."
        assert all(policy >= 0 for policy in self.move_policy), "All move policies should be non-negative."
        assert any(policy > 0 for policy in self.move_policy), "At least one move policy should be positive."

    @classmethod
    def create(
            cls,
            win_rate: float,
            move_policy: Tuple[float, ...],
            playable_moves_bitmap: Optional[Tuple[int, ...]] = None,
            temperature: Optional[float] = None,
            normalize: bool = False
    ) -> EvaluationResult:
        """Creates EvaluationResult by restricting playable moves and using temperature.

        If present, playable_moves_bitmap indicates which values from move_policy should remain (the rest are set to 0).

        If present, temperature modifies the value of the policy, by using following formula x -> x ^ (1 / temp). If
        temp is 0, only highest values remain non-zero.
        """
        if playable_moves_bitmap:
            assert len(move_policy) == len(playable_moves_bitmap), \
                "The move_policy and playable_moves_bitmap need to be the same length"
            move_policy = tuple(
                policy if playable else 0.
                for policy, playable in zip(move_policy, playable_moves_bitmap)
            )

        if temperature is not None:
            if temperature == 0:
                max_policy = max(move_policy)
                move_policy = tuple(
                    1. if policy == max_policy else 0.
                    for policy in move_policy
                )
            else:
                move_policy = tuple(
                    policy ** (1 / temperature)
                    for policy in move_policy
                )

        if normalize:
            policy_sum = sum(move_policy)
            move_policy = tuple(
                policy / policy_sum
                for policy in move_policy
            )

        return cls(win_rate, move_policy)

    def pick_move(self) -> int:
        """Picks move according to move_policy."""
        return random.choices(
            population=range(len(self.move_policy)),
            weights=self.move_policy,
        )[0]

    def pick_best_move(self) -> int:
        """Picks move with the highest move_policy."""
        max_policy = max(self.move_policy)
        return random.choice(
            tuple(
                move_index
                for move_index in range(len(self.move_policy))
                if self.move_policy[move_index] == max_policy
            )
        )

    def __str__(self) -> str:
        return (
                f"win_rate: {self.win_rate:.4f}\n"
                "move_policy: " + " ".join(f"{p:.4f}" for p in self.move_policy)
        )


class Evaluator(ABC):
    """Base class for objects that can evaluate game state."""

    @abstractmethod
    def supports_rules(self, rules: Rules) -> bool:
        """Whether this model supports given rules."""
        raise NotImplementedError()

    @abstractmethod
    def reset_inner_state(self) -> None:
        """Called in order to reset inner state and have a clean game (e.g. for new game)."""
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, state: State) -> EvaluationResult:
        """Evaluates the state."""
        raise NotImplementedError()


class Model(ABC):
    """Base class for playing the game."""

    @abstractmethod
    def supports_rules(self, rules: Rules) -> bool:
        """Whether given rules are supported."""
        raise NotImplementedError()

    @abstractmethod
    def reset_inner_state(self) -> None:
        """Called in order to reset inner state and have a clean game (e.g. for new game)."""
        raise NotImplementedError()

    @abstractmethod
    def play_move(self, state: State) -> MoveOrMoveIndex:
        """Returns move played from a given state."""
        raise NotImplementedError()


class Trainable(ABC):
    """The base class for objects that can be trained using played game."""

    @abstractmethod
    def train(self, training_data: TrainingData) -> TrainingSummary:
        """Training the model with the training data."""
        raise NotImplementedError()


class TrainableEvaluator(Evaluator, Trainable, ABC):
    """Base class for the Evaluator that can be trained."""


class TrainableModel(Model, Trainable, ABC):
    """Base class for the Model that can be trained."""

    @abstractmethod
    def create_training_data_for_game(
            self, result: Result, states: Iterable[State]) -> TrainingData:
        """Creating training data for the played game."""
        raise NotImplementedError()


class TrainingData:
    """Class that contains data used for training.

    Attributes:
        data: The collections of desired State -> EvaluationResult pairs.
    """

    def __init__(self, data: Tuple[Tuple[State, EvaluationResult], ...] = ()):
        self.data = data

    def add(self, training_data: TrainingData) -> None:
        """Add training_data from another object into this one."""
        self.data += training_data.data


class TrainingSummary(ABC):
    """The class that provides summary of a training."""
