from __future__ import annotations

from collections import deque
from typing import NamedTuple

from morphzero.ai.model import TrainingModel
from morphzero.common import print_progress_bar
from morphzero.core.game import Rules, State


class Trainer:
    """Trainer trains the TrainingModel by making it play games against itself."""
    rules: Rules
    model: TrainingModel
    config: Trainer.Config

    def __init__(self,
                 rules: Rules,
                 model: TrainingModel,
                 config: Trainer.Config):
        self.rules = rules
        self.model = model
        self.config = config

    def train(self) -> None:
        engine = self.rules.create_engine()

        for game_index in range(self.config.number_of_games):
            print_progress_bar(game_index + 1, self.config.number_of_games, "Training")

            states = deque[State]()
            state = engine.new_game()
            states.append(state)
            while not state.is_game_over:
                move = self.model.play_move(state)
                state = engine.play_move(state, move)
                states.append(state)

            assert state.result
            self.model.train_from_game(state.result, states)

    class Config(NamedTuple):
        """The configuration for Trainer.

        Attributes:
            number_of_games: The number of games to run.
        """
        number_of_games: int
