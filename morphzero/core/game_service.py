from __future__ import annotations

from morphzero.core.game import Move, State, Engine


class GameService:
    """Keeps track of the game and notifies listeners.

    It tracks progress of the game and notifies the listeners.
    """
    engine: Engine
    state: State
    listeners: list[GameServiceListener]

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.listeners = []
        self.new_game()

    def new_game(self) -> None:
        """Starts new game."""
        self.state = self.engine.new_game()
        for listener in self.listeners:
            listener.on_new_game(self.state)

    def play_move(self, move: Move) -> None:
        """Executes the move and updates current state.

        Raises:
            ValueError: if game is not started or if raised Engine.play_move
        """
        old_state = self.state
        self.state = self.engine.play_move(self.state, move)
        for listener in self.listeners:
            listener.on_move(old_state, move, self.state)

        if self.state.is_game_over:
            for listener in self.listeners:
                listener.on_game_over(self.state)

    def add_listener(self, listener: GameServiceListener) -> None:
        self.listeners.append(listener)

    def remove_listener(self, listener: GameServiceListener) -> None:
        self.listeners.remove(listener)


class GameServiceListener:
    """Listener for the GameService events."""

    def on_new_game(self, state: State) -> None:
        """Called when new game is started."""
        raise NotImplementedError()

    def on_move(self, old_state: State, move: Move, new_state: State) -> None:
        """Called when move is played."""
        raise NotImplementedError()

    def on_game_over(self, state: State) -> None:
        """Called when game is finished."""
        raise NotImplementedError()
