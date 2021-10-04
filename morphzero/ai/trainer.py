from __future__ import annotations

from typing import NamedTuple, Deque, Optional

from morphzero.ai.base import TrainableModel, TrainingData
from morphzero.common import print_progress_bar
from morphzero.core.game import Rules, State


class Trainer:
    """Trainer that trains the TrainingModel by making it play games against itself."""
    rules: Rules
    model: TrainableModel
    config: TrainerConfig

    def __init__(self,
                 rules: Rules,
                 model: TrainableModel,
                 config: TrainerConfig):
        self.rules = rules
        self.model = model
        self.config = config

    def train(self) -> None:
        engine = self.rules.create_engine()
        for batch_index in range(self.config.iterations):
            print(f"Iteration: {batch_index + 1:4d} out of {self.config.iterations}")
            training_data: Optional[TrainingData] = None
            for game_index in range(self.config.simulations):
                print_progress_bar(game_index + 1, self.config.simulations, "Training")

                states = Deque[State]()

                # play a game
                state = engine.new_game()
                states.append(state)
                while not state.is_game_over:
                    move = self.model.play_move(state)
                    state = engine.play_move(state, move)
                    states.append(state)
                assert state.result

                # update training data
                simulation_training_data = self.model.create_training_data_for_game(state.result, states)
                if training_data:
                    training_data.add(simulation_training_data)
                else:
                    training_data = simulation_training_data

            if training_data:
                self.model.train(training_data)


class TrainerConfig(NamedTuple):
    """The configuration for Trainer.

    Attributes:
        iterations: The number of iterations. Model is retrained at the end of each iteration.
        simulations: The number of simulated games played in a iteration.
    """
    iterations: int
    simulations: int
