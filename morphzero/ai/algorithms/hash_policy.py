from __future__ import annotations

import pickle
import random
from typing import Iterable, NamedTuple

from morphzero.ai.algorithms.util import result_for_player, pick_one_of_highest
from morphzero.ai.model import TrainingModel
from morphzero.core.game import State, Rules, Engine, Result, MoveOrMoveIndex


class HashPolicy(dict[State, float]):
    """Stores policy for each observed state, from state.current_player's point of view.

    Unobserved states have policy of 0.5.
    """

    def __missing__(self, key: State) -> float:
        if key.is_game_over:
            assert key.result
            return result_for_player(key.current_player, key.result)
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
    def load(cls, path: str) -> HashPolicy:
        """Loads the HashPolicy from a file with a given path."""
        with open(path, "rb") as f:
            hash_policy = pickle.load(f)
            if not isinstance(hash_policy, cls):
                raise ValueError(f"Type of the loaded object {type(hash_policy)} is not the expected type {cls}")
            return hash_policy


class HashPolicyModel(TrainingModel):
    """Model that learns how to play a game by storing expected policy value for each move."""
    rules: Rules
    engine: Engine
    policy: HashPolicy
    config: HashPolicyModel.Config

    def __init__(self,
                 rules: Rules,
                 policy: HashPolicy,
                 config: HashPolicyModel.Config):
        self.rules = rules
        self.engine = rules.create_engine()
        self.policy = policy
        self.config = config

    def supports_rules(self, rules: Rules) -> bool:
        return self.rules == rules

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
            move_policy = tuple(
                self.evaluate_move(state, move_index)
                for move_index in range(self.rules.number_of_possible_moves())
            )
            move_index = pick_one_of_highest(
                range(len(move_policy)),
                key=lambda i: move_policy[i])
            return move_index

    def evaluate_move(self, state: State, move_index: int) -> float:
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

    def train(self, result: Result, states: Iterable[State]) -> None:
        for state in states:
            self.policy.update_policy(
                state,
                result_for_player(state.current_player, result),
                self.config.learning_rate)

    class Config(NamedTuple):
        learning_rate: float
        exploration_rate: float

        @classmethod
        def create_for_playing(cls) -> HashPolicyModel.Config:
            return cls(learning_rate=0, exploration_rate=0)
