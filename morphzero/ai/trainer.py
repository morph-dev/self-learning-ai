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
        total_games = self.config.iterations * self.config.simulations
        game_index = 0
        for iteration in range(self.config.iterations):
            training_data: Optional[TrainingData] = None
            for simulation in range(self.config.simulations):
                game_index += 1
                print_progress_bar(
                    iteration=game_index,
                    total=total_games,
                    prefix="Training",
                    suffix=f"Iteration: {iteration + 1:d} / {self.config.iterations}, " +
                           f"Simulation: {simulation + 1:d} / {self.config.simulations}\t\t\t"
                )

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
