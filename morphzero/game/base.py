from enum import IntEnum, unique


@unique
class Player(IntEnum):
    NO_PLAYER = 0
    FIRST_PLAYER = 1
    SECOND_PLAYER = -1

    @property
    def other_player(self):
        if self == Player.FIRST_PLAYER:
            return Player.SECOND_PLAYER
        elif self == Player.SECOND_PLAYER:
            return Player.FIRST_PLAYER
        else:
            raise ValueError(f"The {self} doesn't have other player.")


class Rules:
    """
    Properties of the game (board size, winning_condition etc)
    """

    def create_game_engine(self):
        """
        Creates game engine.
        """
        raise NotImplementedError()


class State:
    """
    Uniquely represents the state of the game.
    """

    def __init__(self, current_player, result):
        self._current_player = current_player
        self._result = result

    @property
    def current_player(self):
        """
        Returns player which is responsible for making next action.
        """
        return self._current_player

    @property
    def result(self):
        """
        Returns the result of the game, if game is over. If game is not over, it returns None.
        """
        return self._result

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def is_move_playable(self, state, move):
        """
        Returns whether move is playable from a given state.
        """
        raise NotImplementedError()

    def playable_moves(self, state):
        """
        Returns generator of playable moves, if game is not over. If game is over, it returns None
        and the result should be retrieved from the state.
        """
        raise NotImplementedError()

    def play_move(self, state, move):
        """
        Returns the state of the game that happens after playing given move from the given state.
        """
        raise NotImplementedError()


class GameService:
    def __init__(self, engine):
        self.engine = engine
        self.state = None
        self.listeners = list()

    def new_game(self):
        self.state = self.engine.new_game()
        for listener in self.listeners:
            listener.on_new_game(self.state)

    def play_move(self, move):
        old_state = self.state
        self.state = self.engine.play_move(self.state, move)
        for listener in self.listeners:
            listener.on_move(old_state, move, self.state)

        if self.state.is_game_over:
            for listener in self.listeners:
                listener.on_game_over(self.state)

    def is_move_playable(self, move):
        return self.engine.is_move_playable(self.state, move)

    def add_listener(self, listener):
        self.listeners.append(listener)

    def remove_listener(self, listener):
        self.listeners.remove(listener)

    class Listener:
        def on_new_game(self, state):
            """
            Called when new game is started.
            """
            raise NotImplementedError()

        def on_move(self, old_state, move, new_state):
            """
            Called when move is played.
            """
            raise NotImplementedError()

        def on_game_over(self, state):
            """
            Called when game is finished.
            """
            raise NotImplementedError()
