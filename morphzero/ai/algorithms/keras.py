from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from morphzero.ai.evaluator import Evaluator, EvaluationResult
from morphzero.core.game import Rules, State, Engine


class KerasEvaluator(Evaluator):
    rules: Rules
    config: KerasEvaluator.Config
    engine: Engine
    model: tf.keras.Model

    def __init__(self,
                 rules: Rules,
                 config: KerasEvaluator.Config):
        self.rules = rules
        self.config = config

        self.engine = rules.create_engine()
        self.model = self.create_model()

    def supports_rules(self, rules: Rules) -> bool:
        return self.rules == rules

    def evaluate(self, state: State) -> EvaluationResult:
        win_rate_tensor, move_policy_tensor = self.model(
            tf.convert_to_tensor([state.to_training_data()]),
            training=self.config.training
        )
        move_policy = move_policy_tensor[0].numpy()
        for move_index, playable in enumerate(self.engine.playable_moves_bitmap(state)):
            if not playable:
                move_policy[move_index] = 0
        return EvaluationResult.normalize_and_create(
            win_rate_tensor[0][0].numpy(),
            tuple(move_policy_tensor[0].numpy()),
        )

    def train(self, learning_data: dict[State, EvaluationResult]) -> None:
        states, evaluation_results = zip(*list(learning_data.items()))
        win_rates, move_policies = zip(*evaluation_results)
        inputs = tf.convert_to_tensor([state.to_training_data() for state in states])
        outputs = [
            tf.convert_to_tensor([[win_rate] for win_rate in win_rates]),
            tf.convert_to_tensor([move_policy for move_policy in move_policies]),
        ]

        self.model.fit(
            inputs,
            outputs,
            initial_epoch=0,
            epochs=self.config.epochs,
            callbacks=[
                # keras.callbacks.ModelCheckpoint(
                #     filepath='path/to/my/model_{epoch}'),
            ],
            shuffle=True,
            verbose=self.config.verbose,
        )

    def create_model(self) -> tf.keras.Model:
        """Creates a model that has State.to_training_data as input and win_rate and move_policy as output."""
        raise NotImplementedError()

    @dataclass(frozen=True)
    class Config:
        training: bool
        verbose: int
        learning_rate: float
        epochs: int
