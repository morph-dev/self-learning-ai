from __future__ import annotations

import pickle
import random
from typing import NamedTuple, Callable, Optional, Iterable, Dict

from morphzero.ai.algorithms.util import result_for_player
from morphzero.ai.base import TrainableEvaluator, TrainableModel, EvaluationResult, TrainingSummary, TrainingData
from morphzero.core.game import State, Rules, Engine, MoveOrMoveIndex, Result


class StateHashPolicy(Dict[State, float]):
    """Stores policy for each observed state, from state.current_player's point of view.

    Unobserved states have policy of 0.5.
    """

    def __missing__(self, state: State) -> float:
        if state.is_game_over:
            assert state.result
            return result_for_player(state.current_player, state.result)
        else:
            return 0.5

    def update_policy(self,
                      state: State,
                      desired_policy: float,
                      learning_rate: float) -> None:
        """Updates state policy.

        Args:
            state: State to update.
            desired_policy: New desired policy.
            learning_rate: The ration [0, 1] of difference to move towards desired policy.
        """
        policy = self[state]
        policy += (desired_policy - policy) * learning_rate
        self[state] = policy

    def store(self, path: str) -> None:
        """Stores the HashPolicy in a file with a given path."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> StateHashPolicy:
        """Loads the HashPolicy from a file with a given path."""
        with open(path, "rb") as f:
            hash_policy = pickle.load(f)
            if not isinstance(hash_policy, cls):
                raise ValueError(f"Type of the loaded object {type(hash_policy)} is not the expected type {cls}")
            return hash_policy


class HashPolicyConfig(NamedTuple):
    learning_rate: float
    exploration_rate: float
    valid_move_min_value: float = 0.000001
    temperature: float = 0.  # play the best move

    @classmethod
    def create_for_playing(cls) -> HashPolicyConfig:
        return cls(learning_rate=0., exploration_rate=0., temperature=0.)


class HashPolicy(TrainableEvaluator, TrainableModel):
    """Evaluator and Model that learns how to play a game by storing expected policy value for each move."""
    rules: Rules
    engine: Engine
    policy: StateHashPolicy
    config: HashPolicyConfig

    def __init__(self,
                 rules: Rules,
                 policy: StateHashPolicy,
                 config: HashPolicyConfig):
        self.rules = rules
        self.engine = rules.create_engine()
        self.policy = policy
        self.config = config

    def supports_rules(self, rules: Rules) -> bool:
        return rules == self.rules

    def reset_inner_state(self) -> None:
        pass

    def evaluate(self, state: State) -> EvaluationResult:
        return EvaluationResult.create(
            win_rate=self.policy[state],
            move_policy=tuple(
                self.evaluate_move(state, move_index)
                for move_index in range(self.rules.number_of_possible_moves())
            ),
            temperature=self.config.temperature,
        )

    def play_move(self, state: State) -> MoveOrMoveIndex:
        if state.is_game_over:
            raise ValueError("Game is over.")
        if random.random() < self.config.exploration_rate:
            playable_moves = tuple(
                move
                for move in self.engine.playable_moves(state)
                if not move.resign
            )
            return random.choice(playable_moves)
        else:
            return self.evaluate(state).pick_move()

    def evaluate_move(self, state: State, move_index: int) -> float:
        """Evaluates the move for the given state."""
        if move_index == self.engine.get_move_index_for_resign():
            return 0
        if not self.engine.is_move_playable(state, move_index):
            return 0
        next_state = self.engine.play_move(state, move_index)
        next_state_policy = self.policy[next_state]
        if next_state.current_player == state.current_player:
            return max(next_state_policy, self.config.valid_move_min_value)
        elif next_state.current_player == state.current_player.other_player:
            return max(1 - next_state_policy, self.config.valid_move_min_value)
        else:
            raise ValueError(f"Unexpected next_state.current_player: {next_state.current_player}")

    def create_training_data_for_game(
            self, result: Result, states: Iterable[State]) -> TrainingData:
        return TrainingData(
            tuple(
                (
                    state,
                    EvaluationResult.create(
                        win_rate=result_for_player(state.current_player, result),
                        move_policy=self.engine.playable_moves_bitmap(state),
                    )
                )
                for state in states
                if not state.is_game_over
            )
        )

    def train(self, training_data: TrainingData) -> TrainingSummary:
        for state, desired_evaluation_result in training_data.data:
            self.policy.update_policy(
                state,
                desired_evaluation_result.win_rate,
                self.config.learning_rate)
        return TrainingSummary()

    @classmethod
    def factory(cls, path: str, config: Optional[HashPolicyConfig] = None) -> Callable[[Rules], HashPolicy]:
        return lambda rules: cls(
            rules,
            StateHashPolicy.load(path),
            config if config else HashPolicyConfig.create_for_playing())
