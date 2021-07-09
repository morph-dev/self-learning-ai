from __future__ import annotations

from collections import deque
from typing import NamedTuple, Optional

from morphzero.ai.model import TrainingModel
from morphzero.common import board_to_string
from morphzero.core.game import Rules, State


class Trainer:
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
        print_ratio_increment = self.config.print_ratio_increment
        next_print_ratio = 0.

        for game_index in range(self.config.number_of_games):
            if print_ratio_increment:
                if round(next_print_ratio * self.config.number_of_games) <= game_index:
                    print(f"Training {round(next_print_ratio * 100)}%")
                    next_print_ratio += print_ratio_increment

            states = deque[State]()
            state = engine.new_game()
            states.append(state)
            while not state.is_game_over:
                move = self.model.play_move(state)
                state = engine.play_move(state, move)
                states.append(state)

            assert state.result
            self.model.train(state.result, states)

    class Config(NamedTuple):
        number_of_games: int
        print_ratio_increment: Optional[float] = None
