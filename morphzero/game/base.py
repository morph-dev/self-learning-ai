from enum import IntEnum, unique

@unique
class Player(IntEnum):
    NO_PLAYER = 0
    FIRST_PLAYER = 1
    SECOND_PLAYER = -1

    def other_player(self):
        if self is Player.FIRST_PLAYER:
            return Player.SECOND_PLAYER
        elif self is Player.SECOND_PLAYER:
            return Player.FIRST_PLAYER
        else:
            raise ValueError(f"The {self} doesn't have other player.")

class Rules:
    """
    Properties of the game (board size, player names, etc)
    """
    pass

class Move:
    """
    Describes possible move.
    """
    def __init__(self, key):
        self._key = key
    def __hash__(self):
        return hash(self._key)
    def __eq__(self, other):
        return isinstance(other, Move) and self._key == other._key
    def __repr__(self):
        return str(self._key)

class State:
    """
    Uniquely represents the state of the game.
    """

    @property
    def current_player(self):
        """
        Returns player which is responsible for making next action.
        """
        pass

    @property
    def result(self):
        """
        Returns the result of the game, if game is over. If game is not over, it returns None.
        """
        pass

    @property
    def is_game_over(self):
        """
        Returns whether game is over.
        """
        return self.result is not None

    def key(self):
        """
        Returns key of the state. It should contain minimal info required for hashing and
        comparison. Type of the key can be anything that is hashable and comparable.
        """
    def __hash__(self):
        return hash(self.key())
    def __eq__(self, other):
        return isinstance(other, State) and self.key() == other.key()

class GameEngine:
    def __init__(self, rules):
        self.rules = rules

    def new_game(self):
        """
        Returns the State the corresponds to the state of the new game according to rules.
        """
        pass

    def is_move_playable(self, state, move):
        """
        Returns whether move is playable from a given state.
        """
        pass

    def playable_moves(self, state):
        """
        Returns generator of playable moves, if game is not over. If game is over, it returns None
        and the result should be retrieved from the state.
        """
        pass

    def play_move(self, state, move):
        """
        Returns the state of the game that happens after playing given move from the given state.
        """
        pass
