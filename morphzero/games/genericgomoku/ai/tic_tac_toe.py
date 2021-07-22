from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras

from morphzero.ai.algorithms.keras import KerasEvaluator
from morphzero.games.genericgomoku.game import GenericGomokuRules


class TicTacToeKeras(KerasEvaluator):
    rules: GenericGomokuRules
    config: TicTacToeKeras.Config

    def __init__(self, config: TicTacToeKeras.Config):
        rules = GenericGomokuRules.create_tic_tac_toe_rules()
        super().__init__(rules, config)

    def create_model(self) -> keras.Model:
        width, height = self.rules.board_size
        board_input = keras.Input(shape=(width, height), name="game_board", dtype=tf.float32)
        reshaped = keras.layers.Reshape((width, height, 1))(board_input)
        conv1 = keras.layers.Conv2D(
            filters=self.config.filters,
            kernel_size=self.config.kernel_size,
            activation="tanh",
            padding="same",
            data_format="channels_last")(reshaped)
        conv2 = keras.layers.Conv2D(
            filters=self.config.filters,
            kernel_size=self.config.kernel_size,
            activation="tanh",
            padding="same",
            data_format="channels_last")(conv1)
        conv2_flat = keras.layers.Flatten()(conv2)
        mid_layer = keras.layers.Dense(
            self.config.mid_layer_size,
            activation="tanh")(conv2_flat)
        win_rate = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="win_rate")(mid_layer)
        move_policy = keras.layers.Dense(
            self.rules.number_of_possible_moves(),
            activation="softmax",
            name="move_policy")(mid_layer)
        model = keras.Model(inputs=board_input, outputs=[win_rate, move_policy])
        model.compile(
            loss=[
                'mean_squared_error',
                'categorical_crossentropy',
            ],
            optimizer=keras.optimizers.Adam(self.config.learning_rate))
        return model

    @dataclass(frozen=True)
    class Config(KerasEvaluator.Config):
        filters: int
        mid_layer_size: int
        kernel_size: tuple[int, int] = (3, 3)


# class dotdict(dict):
#     def __getattr__(self, name):
#         return self[name]
#
#
# class OthelloNNet():
#     def __init__(self, args):
#         # game params
#         self.board_x, self.board_y = 3, 3
#         self.action_size = 10
#         self.args = args
#
#         # Neural Net
#         self.input_boards = keras.Input(shape=(self.board_x, self.board_y))  # s: batch_size x board_x x board_y
#
#         x_image = keras.layers.Reshape((self.board_x, self.board_y, 1))(
#             self.input_boards)  # batch_size  x board_x x board_y x 1
#         h_conv1 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=3)(
#             keras.layers.Conv2D(args.num_channels, 3, padding='same', use_bias=False)(
#                 x_image)))  # batch_size  x board_x x board_y x num_channels
#         h_conv2 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=3)(
#             keras.layers.Conv2D(args.num_channels, 3, padding='same', use_bias=False)(
#                 h_conv1)))  # batch_size  x board_x x board_y x num_channels
#         h_conv3 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=3)(
#             keras.layers.Conv2D(args.num_channels, 3, padding='same', use_bias=False)(
#                 h_conv2)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
#         h_conv4 = keras.layers.Activation('relu')(keras.layers.BatchNormalization(axis=3)(
#             keras.layers.Conv2D(args.num_channels, 3, padding='valid', use_bias=False)(
#                 h_conv3)))  # batch_size  x (board_x-4) x (board_y-4) x num_channels
#         h_conv4_flat = keras.layers.Flatten()(h_conv4)
#         s_fc1 = keras.layers.Dropout(args.dropout)(keras.layers.Activation('relu')(
#             keras.layers.BatchNormalization(axis=1)(
#                 keras.layers.Dense(1024, use_bias=False)(h_conv4_flat))))  # batch_size x 1024
#         s_fc2 = keras.layers.Dropout(args.dropout)(keras.layers.Activation('relu')(
#             keras.layers.BatchNormalization(axis=1)(
#                 keras.layers.Dense(512, use_bias=False)(s_fc1))))  # batch_size x 1024
#         self.pi = keras.layers.Dense(self.action_size, activation='softmax', name='pi')(
#             s_fc2)  # batch_size x self.action_size
#         self.v = keras.layers.Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1
#
#         self.model = keras.Model(inputs=self.input_boards, outputs=[self.pi, self.v])
#         self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
#                            optimizer=keras.optimizers.Adam(args.lr))
#
#
# args = dotdict({
#     'lr': 0.001,
#     'dropout': 0.3,
#     'epochs': 10,
#     'batch_size': 64,
#     'cuda': False,
#     'num_channels': 512,
# })