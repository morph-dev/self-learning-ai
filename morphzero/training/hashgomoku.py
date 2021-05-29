import math
import random

from morphzero.game.base import Player
from morphzero.training.base import Model, Trainer


class HashGomokuModel(Model):
    """
    Model that learns to play GenericGomoku game by having a policy value for each different state.
    It's not able to make a decision/prediction for a state that has never seen before, but it works
    quite well for games with small number of possible states, e.g. TicTakToe.
    """

    def __init__(self):
        self.state_policy = dict()

    def _get_move_value(self, game_engine, state, move):
        """
        Smaller returned value means better move.
        """
        new_state = game_engine.play_move(state, move)
        policy_value = self.state_policy.get(new_state, math.inf)
        return abs(state.current_player - policy_value)

    def play_move(self, game_engine, state):
        moves = [
            (move, self._get_move_value(game_engine, state, move))
            for move in game_engine.playable_moves(state)
        ]

        best_move_value = min((move_value for (move, move_value) in moves))
        best_moves = [move for (move, move_value) in moves if move_value <= best_move_value]
        return random.choice(best_moves)


class HashGomokuTrainer(Trainer):
    """
    Trainer for the HashGomokuModel.
    """

    def __init__(self, game_engine, learning_rate, exploration_rate, model=None):
        super().__init__(game_engine)
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.model = model if model is not None else HashGomokuModel()
        self.game_states = []

    def on_game_start(self):
        self.game_states = []

    def play_move(self, state):
        self.game_states.append(state)
        if random.random() < self.exploration_rate:
            return random.choice(list(self.game_engine.playable_moves(state)))
        return self.model.play_move(self.game_engine, state)

    def on_game_end(self, state):
        self.game_states.append(state)
        score = state.result.winner
        for state in reversed(self.game_states):
            state_value = self.model.state_policy.get(state, Player.NO_PLAYER)
            state_value += (score - state_value) * self.learning_rate
            score = state_value
            self.model.state_policy[state] = score
