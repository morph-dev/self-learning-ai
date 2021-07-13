from __future__ import annotations

import math
import random
import time
from collections import deque, Iterable
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

from morphzero.ai.algorithms.util import pick_one_of_highest, result_for_player
from morphzero.ai.evaluator import Evaluator, EvaluationResult
from morphzero.core.game import Rules, State, Engine, Result, Move


class PureMonteCarloTreeSearch(Evaluator):
    """The Ai that evaluates the state of a game based on Pure MonteCarloTreeSearch.

    The implementation is strongly based on https://en.wikipedia.org/wiki/Monte_Carlo_tree_search.

    There is no heuristics or predictions involved. Playout of the game runs until end of game is reached.

    Attributes:
        rules: Rules of the game being played.
        config: Learning configuration of the game.
        engine: Engine created from rules.
        nodes: Maps State to Node associated with it for quick access.
        discovered_states: States that were visited during playouts. Used in order to keep only one copy of the same
            state.
    """
    rules: Rules
    config: PureMonteCarloTreeSearch.Config

    engine: Engine
    nodes: dict[State, _Node]
    discovered_states: dict[State, State]

    def __init__(self, rules: Rules, config: PureMonteCarloTreeSearch.Config):
        self.rules = rules
        self.config = config
        self.engine = self.rules.create_engine()
        self.nodes = dict()

        self.discovered_states = dict()

    def supports_rules(self, rules: Rules) -> bool:
        return self.rules == rules

    def evaluate(self, state: State) -> EvaluationResult:
        assert not state.is_game_over, "Can't evaluate Game Over state."
        start_time_sec = time.time()
        for simulation_index in range(self.config.number_of_simulations):
            elapsed_time_sec = time.time() - start_time_sec
            if self.config.max_time_sec and elapsed_time_sec > self.config.max_time_sec:
                print(f"Only {simulation_index} out of {self.config.number_of_simulations}.")
                break
            self.simulation(state)

        node = self.nodes[state]
        move_policy_dict = node.move_policy_dict()
        move_policy = [0.] * self.rules.number_of_possible_moves()
        for move_index in move_policy_dict:
            move_policy[move_index] = move_policy_dict[move_index]

        return EvaluationResult(node.best_move().win_ratio, tuple(move_policy))

    def simulation(self, state: State) -> None:
        """Runs one MonteCarloTreeSearch simulation.

        It has following stages:
            Selection: Iterates over discovered Nodes in the game tree, until new Node is discovered or game is over.

            Expansion: Expands discovered Node (if applicable)

            Rollout: Random playout of the game from the Discovered Node until game is over.

            Backpropagation: Update nodes within selection with the outcome of the rollout stage.
        """
        node_moves = deque[tuple[_Node, _MoveInfo]]()
        # Selection & Expansion
        expanded = False
        while not state.is_game_over and not expanded:
            if state in self.nodes:
                node = self.nodes[state]
            else:
                node = _Node(state)
                self.nodes[state] = node
                node.expand(self.engine, self.discovered_states)
                expanded = True
            move_info = node.play_move(self.config.exploration_rate)
            node_moves.append((node, move_info))
            state = move_info.next_state

        # Rollout
        while not state.is_game_over:
            moves = [
                move
                for move in self.engine.playable_moves(state)
                if not move.resign
            ]
            move = random.choice(moves)
            state = self.engine.play_move(state, move)
        # Backpropagation
        assert state.result
        for (node, move_info) in node_moves:
            node.update(state.result, move_info)

    def train(self, result: Result, states: Iterable[State]) -> None:
        raise TypeError("Training on played games not supported.")

    class Config(NamedTuple):
        """The configuration for the PureMonteCarloTreeSearch."""
        number_of_simulations: int
        exploration_rate: float = 1.4
        max_time_sec: Optional[float] = 1


class _Node:
    """Represents and stores info associated with one node in the game tree.

    Attributes:
        state: The Game state.
        total_exploration_count: The Number of times we explored this node.
        moves: The tuple of _MoveInfo that represent edges between nodes in the game tree.
    """
    state: State
    total_exploration_count: int
    moves: tuple[_MoveInfo, ...]

    def __init__(self, state: State):
        assert not state.is_game_over
        self.state = state
        self.total_exploration_count = 0
        self.moves = ()

    def expand(self, engine: Engine, discovered_states: dict[State, State]) -> None:
        """Called in order to expand node to it's children in the game tree."""
        def create_move_info(move: Move) -> _MoveInfo:
            next_state = engine.play_move(self.state, move)
            if next_state in discovered_states:
                next_state = discovered_states[next_state]
            else:
                discovered_states[next_state] = next_state
            return _MoveInfo(move.move_index, next_state)

        self.moves = tuple(
            create_move_info(move)
            for move in engine.playable_moves(self.state)
        )

    def best_move(self) -> _MoveInfo:
        """Returns one of the moves with the highest win_rate."""
        return self.play_move(exploration_rate=0)

    def move_policy_dict(self) -> dict[int, float]:
        """Returns move_index -> win_rate dictionary."""
        return {
            move_info.move_index: move_info.win_ratio
            for move_info in self.moves
        }

    def play_move(self, exploration_rate: float) -> _MoveInfo:
        """Plays the move according to exploration_rate.

        Args:
            exploration_rate: Lower value means that win_rate of moves matters more than exploring other moves.
        """
        assert not self.state.is_game_over, "Can't play a move from game_over state."
        assert self.moves, "Moves never initialized."

        def uct(move_info: _MoveInfo) -> float:
            return self.uct(move_info, exploration_rate)

        return pick_one_of_highest(self.moves, uct)

    def uct(self, move_info: _MoveInfo, exploration_rate: float) -> float:
        """Upper Confidence bounds applied to Trees.

        Returns the score for exploring given move.
        """
        if self.total_exploration_count == 0:
            # No moves were explored.
            return 1

        expansion_value = move_info.win_ratio
        exploration_value = math.sqrt(math.log(1 + self.total_exploration_count) / (1 + move_info.exploration_count))
        return expansion_value + exploration_rate * exploration_value

    def update(self, result: Result, move_info: _MoveInfo) -> None:
        """Updates win_rate based on the result of the simulated game."""
        result_value = result_for_player(self.state.current_player, result)
        self.total_exploration_count += 1
        move_info.exploration_count += 1
        move_info.exploration_wins += result_value

    def __str__(self) -> str:
        return f"state:{self.state};total_exploration_count:{self.total_exploration_count};"


@dataclass
class _MoveInfo:
    """Stores info regarding playing moves.

    Attributes:
        move_index: The index of the move in the original state.
        next_state: The state achieved after playing the move.
        exploration_wins: Number of simulations the ended in a win for state.current_player. Draw adds 0.5 to it.
        exploration_count: Number of simulations for this move.
    """
    move_index: int
    next_state: State = field(repr=False)
    exploration_wins: float = 0
    exploration_count: int = 0

    @property
    def win_ratio(self) -> float:
        """Returns exploration_win / exploration_count.

        If exploration_count is 0, returns 0.5.
        """
        if self.exploration_count == 0:
            return 0.5
        return self.exploration_wins / self.exploration_count
