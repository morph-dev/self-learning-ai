from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import tensorflow as tf

from morphzero.ai.base import TrainableEvaluator, EvaluationResult, TrainingData, TrainingSummary
from morphzero.core.game import Rules, State, Engine


@dataclass(frozen=True)
class KerasEvaluatorConfig:
    training: bool
    verbose: int
    learning_rate: float
    epochs: int


class KerasEvaluator(TrainableEvaluator):
    rules: Rules
    config: KerasEvaluatorConfig
    engine: Engine
    model: tf.keras.Model

    def __init__(self,
                 rules: Rules,
                 config: KerasEvaluatorConfig):
        self.rules = rules
        self.config = config

        self.engine = rules.create_engine()
        self.model = self.create_model()

    def supports_rules(self, rules: Rules) -> bool:
        return self.rules == rules

    def reset_inner_state(self) -> None:
        pass

    def evaluate(self, state: State) -> EvaluationResult:
        win_rate_tensor, move_policy_tensor = self.model(
            tf.convert_to_tensor([state.to_training_data()]),
            training=self.config.training
        )
        return EvaluationResult.create(
            win_rate=win_rate_tensor[0][0].numpy(),
            move_policy=tuple(move_policy_tensor[0].numpy()),
            playable_moves_bitmap=self.engine.playable_moves_bitmap(state),
            normalize=True,
        )

    def train(self, training_data: TrainingData) -> KerasTrainingSummary:
        inputs = tf.convert_to_tensor(
            [
                state.to_training_data()
                for state, _ in training_data.data
            ]
        )
        outputs = [
            tf.convert_to_tensor(
                [
                    [evaluation_result.win_rate]
                    for _, evaluation_result in training_data.data
                ]
            ),
            tf.convert_to_tensor(
                [
                    evaluation_result.move_policy
                    for _, evaluation_result in training_data.data
                ]
            ),
        ]
        history = self.model.fit(
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
        return KerasTrainingSummary(history)

    @abstractmethod
    def create_model(self) -> tf.keras.Model:
        """Creates a model that has State.to_training_data as input and win_rate and move_policy as output."""
        raise NotImplementedError()


@dataclass
class KerasTrainingSummary(TrainingSummary):
    keras_history: tf.keras.callbacks.History
