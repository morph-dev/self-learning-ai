
class Rules:
    """
    Properties of the game (board size, player names, etc)
    """
    pass

class State:
    """
    Uniquely represents the state of the game.
    """

    def result(self):
        """
        Returns the current result of the game. If game is not finished, it returns None.
        """
        pass
    def isGameOver(self):
        """
        Returns whether game is over.
        """
        return result is None

    def key(self):
        """
        Returns key of the state. It should contain minimal info required for hashing and
        comparison. Type of the key can be anything that is hashable and comparable.
        """
    def __hash__(self):
        return hash(self.key())
    def __eq__(self, other):
        return isinstance(other, State) and self.key() == other.key()

    def serialize(self):
        """
        Returns string that uniquely represents the state and can be used for deserialization.
        """
        pass

    @classmethod
    def deserialize(cls, serialized):
        """
        Creates State from serialized string.
        """
        pass

class Move:
    """
    Describes possible move.
    """
    def __init__(self, key):
        self.key = key
    def __hash__(self):
        return hash(self.key)
    def __eq__(self, other):
        return isinstance(other, Move) and self.key == other.key

class GameEngine:
    def __init__(self, rules):
        self.rules = rules

    def newGame(self):
        pass

    def availableMoves(self, state):
        """
        Returns list of available moves. If list is empty, game is over and the result should be
        retrieved from the state.
        """
        pass

    def playMove(self, state, move):
        """
        Returns the state of the game that happens after playing given move from the given state.
        """
        pass
