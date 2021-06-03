import pickle

from morphzero.game.ai_player import AiPlayer


class Model(AiPlayer):
    """
    Abstract class for playing and the game.
    """

    def play_move(self, game_engine, state):
        """
        Returns move to play from a given state.
        """
        raise NotImplementedError()

    def serialize(self, path):
        """
        Writes the model into given file path.
        """
        f = open(path, "wb")
        pickle.dump(self, f)
        f.close()

    @classmethod
    def deserialize(cls, path):
        """
        Reads the model from given file path.
        """
        f = open(path, "rb")
        model = pickle.load(f)
        f.close()
        return model


class Trainer:
    """
    Abstract trainer class for training the model.
    """

    def __init__(self, game_engine):
        self.game_engine = game_engine

    def on_game_start(self):
        raise NotImplementedError()

    def play_move(self, state):
        raise NotImplementedError()

    def on_game_end(self, state):
        raise NotImplementedError()
