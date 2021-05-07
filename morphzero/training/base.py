import pickle

class Model:
    """
    Abstract class for playing and the game.
    """
    def play_move(self, game_engine, state):
        """
        Returns move to play from a given state.
        """
        pass

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
        pass
    def play_move(self, state):
        pass
    def on_game_end(self, state):
        pass
