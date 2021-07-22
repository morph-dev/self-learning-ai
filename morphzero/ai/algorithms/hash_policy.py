from __future__ import annotations

import pickle
import random
from typing import NamedTuple, Callable, Optional, Iterable

from morphzero.ai.algorithms.util import result_for_player, pick_one_index_with_highest_value
from morphzero.ai.evaluator import Evaluator, EvaluationResult
from morphzero.ai.model import TrainingModel
from morphzero.core.game import State, Rules, Engine, MoveOrMoveIndex, Result


class StateHashPolicy(dict[State, float]):
    """Stores policy for each observed state, from state.current_player's point of view.

    Unobserved states have policy of 0.5.
    """

    def __missing__(self, state: State) -> float:
        if state.is_game_over:
            assert state.result
            self[state] = result_for_player(state.current_player, state.result)
        else:
            self[state] = 0.5
        return self[state]

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


class HashPolicy(Evaluator, TrainingModel):
    """Evaluator and Model that learns how to play a game by storing expected policy value for each move."""
    rules: Rules
    engine: Engine
    policy: StateHashPolicy
    config: HashPolicy.Config

    def __init__(self,
                 rules: Rules,
                 policy: StateHashPolicy,
                 config: HashPolicy.Config):
        self.rules = rules
        self.engine = rules.create_engine()
        self.policy = policy
        self.config = config

    def supports_rules(self, rules: Rules) -> bool:
        return rules == self.rules

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
            move_policy = self.evaluate(state).move_policy
            move_index = pick_one_index_with_highest_value(move_policy)
            return move_index

    def evaluate(self, state: State) -> EvaluationResult:
        return EvaluationResult.normalize_and_create(
            win_rate=self.policy[state],
            move_policy=tuple(
                self.evaluate_move(state, move_index)
                for move_index in range(self.rules.number_of_possible_moves())
            )
        )

    def evaluate_move(self, state: State, move_index: int) -> float:
        """Evaluates the move for the given state."""
        if not self.engine.is_move_playable(state, move_index):
            return 0
        next_state = self.engine.play_move(state, move_index)
        next_state_policy = self.policy[next_state]
        if next_state.current_player == state.current_player:
            return next_state_policy
        elif next_state.current_player == state.current_player.other_player:
            return 1 - next_state_policy
        else:
            raise ValueError(f"Unexpected next_state.current_player: {next_state.current_player}")

    def train_from_game(self, result: Result, states: Iterable[State]) -> None:
        self.train({
            state: EvaluationResult.normalize_and_create(
                result_for_player(state.current_player, result),
                tuple())
            for state in states
        })

    def train(self, learning_data: dict[State, EvaluationResult]) -> None:
        for state in learning_data:
            self.policy.update_policy(
                state,
                learning_data[state].win_rate,
                self.config.learning_rate)

    @classmethod
    def factory(cls, path: str, config: Optional[HashPolicy.Config] = None) -> Callable[[Rules], HashPolicy]:
        return lambda rules: cls(
            rules,
            StateHashPolicy.load(path),
            config if config else HashPolicy.Config.create_for_playing())

    class Config(NamedTuple):
        learning_rate: float
        exploration_rate: float

        @classmethod
        def create_for_playing(cls) -> HashPolicy.Config:
            return cls(learning_rate=0, exploration_rate=0)
