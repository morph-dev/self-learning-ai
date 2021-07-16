from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple, Iterable, Optional, Callable

from morphzero.ai.algorithms.util import pick_one_with_highest_value, result_for_player
from morphzero.ai.evaluator import Evaluator, EvaluationResult
from morphzero.ai.model import TrainingModel
from morphzero.core.game import State, Result, Rules, Engine


class MonteCarloTreeSearch(TrainingModel):
    """The Monte Carlo Tree Search algorithm that uses other evaluators as base.

    Attributes:
        rules: The rules of the game.
        engine: The game engine.
        evaluator: The evaluator used for evaluating states visited for the first time.
        config: The Monte Carlo Tree Search configuration.
        nodes: The State to _Node mapping. It initializes _Node-s if they don't already exist.
    """
    rules: Rules
    engine: Engine
    evaluator: Evaluator
    config: MonteCarloTreeSearch.Config
    nodes: _StateToNodeDefaultDict

    def __init__(self, rules: Rules, evaluator: Evaluator, config: MonteCarloTreeSearch.Config):
        if not evaluator.supports_rules(rules):
            raise ValueError("Evaluator doesn't support rules")
        self.rules = rules
        self.engine = self.rules.create_engine()
        self.evaluator = evaluator
        self.config = config

        self.nodes = _StateToNodeDefaultDict(self)

    def supports_rules(self, rules: Rules) -> bool:
        return self.rules == rules

    def play_move(self, state: State) -> int:
        assert not state.is_game_over, "Can't play a move when game is already over"
        start_time_sec = time.time()
        for simulation_index in range(self.config.number_of_simulations):
            elapsed_time_sec = time.time() - start_time_sec
            if self.config.max_time_sec and elapsed_time_sec > self.config.max_time_sec:
                print(f"Only {simulation_index} out of {self.config.number_of_simulations} simulations.")
                break
            self.simulation(state)

        move_indexes = deque[int]()
        move_policies = deque[float]()
        for move_index, move_policy in self.get_move_policy_dict(state).items():
            move_indexes.append(move_index)
            move_policies.append(move_policy)
        return random.choices(move_indexes, move_policies)[0]

    def train_from_game(self, result: Result, states: Iterable[State]) -> None:
        """Passes the training information to the evaluator."""

        def get_desired_evaluation_result(state: State) -> EvaluationResult:
            move_policy_dict = self.get_move_policy_dict(state)
            move_policy = [0.] * self.rules.number_of_possible_moves()
            for index, policy in move_policy_dict.items():
                move_policy[index] = policy
            return EvaluationResult(
                win_rate=result_for_player(state.current_player, result),
                move_policy=tuple(move_policy))

        self.evaluator.train({
            state: get_desired_evaluation_result(state)
            for state in states
        })

    def get_move_policy_dict(self, state: State) -> dict[int, float]:
        """Returns move policy as a dictionary for a given state."""
        return self.nodes[state].get_move_policy_dict()

    def simulation(self, root_state: State) -> None:
        """Runs one MonteCarloTreeSearch simulation.

        Run has 4 stages:
            Selection: Nodes are explored until end of the game is reached or non-expanded node is reached.

            Expansion: If non-expanded node is reach, expand it. It uses evaluator to predict result for the state for
                move_policy for each move.

            Rollout: We don't perform rollout. Instead we use result predicted by evaluator.

            Backpropagation: Update all visited nodes with the result.
        """
        # Selection
        node = self.nodes[root_state]
        node_moves = deque[tuple[_Node, _MoveInfo]]()
        while not node.state.is_game_over and node.expanded:
            move_info = node.play_move()
            node_moves.append((node, move_info))
            node = move_info.next_node

        # Expansion
        if not node.state.is_game_over:
            assert not node.expanded
            node.expand()

        # Rollout
        # We don't do a rollout. Instead we use result predicted by evaluator.
        result_per_player = {
            node.state.current_player: node.result_prediction,
            node.state.current_player.other_player: 1 - node.result_prediction,
        }

        # Backpropagation
        for node, move_info in node_moves:
            node.update(move_info, result_per_player[node.state.current_player])

    @classmethod
    def factory(cls,
                evaluator_factory: Callable[[Rules], Evaluator],
                config: MonteCarloTreeSearch.Config) -> Callable[[Rules], MonteCarloTreeSearch]:
        return lambda rules: MonteCarloTreeSearch(rules, evaluator_factory(rules), config)

    class Config(NamedTuple):
        """The configuration of the MonteCarloTreeSearch algorithm.

        Attributes:
            number_of_simulations: The number of simulations to run.
            exploration_rate: The Exploration rate of the algorithm.
            temperature: The lower the temperature, higher policy value for moves with higher exploration count. For
                temperature 0, only moves with maximum exploration count have non-zero value.
            max_time_sec: It will stop simulations if it is running longer that this (optional).
        """
        number_of_simulations: int
        exploration_rate: float = 1.4
        temperature: float = 1
        max_time_sec: Optional[float] = 1


class _Node:
    """Represents the node in the game tree.

    Nodes for EndGame states can't be expanded.

    Only expanded nodes can be used for playing next moves.

    Non-expanded nodes are the ones that are not yet explored (but we discovered them).
    Node gets expanded first time it's explored. This causes it to be evaluated by evaluator (whose results are stored)
    and makes all child nodes discovered as well (if they weren't previously).

    Attributes:
        state: The state of the game.
        mcts: MonteCarloTreeSearch class used.
        expanded_info: The information useful only once Node is expanded.
    """
    state: State
    mcts: MonteCarloTreeSearch
    expanded_info: Optional[_Node.ExpandedInfo]

    @dataclass
    class ExpandedInfo:
        """Information available once Node has been expanded.

        Attributes:
            total_exploration_count: The number of times this node has been explored.
            result_prediction: The result predicted for the state by the evaluator.
            moves: The tuple of _MoveInfo for playable moves.
        """
        total_exploration_count: int
        result_prediction: float
        moves: tuple[_MoveInfo, ...]

    def __init__(self, state: State, mcts: MonteCarloTreeSearch):
        self.state = state
        self.mcts = mcts
        self.expanded_info = None

    @property
    def expanded(self) -> bool:
        """Whether node was already expanded."""
        return self.expanded_info is not None

    @property
    def expandable(self) -> bool:
        """Whether node is expandable."""
        return not self.state.is_game_over

    def expand(self) -> None:
        """Expands the node. See class details for clarification on the expanded state."""
        if not self.expandable:
            return
        evaluation_result = self.mcts.evaluator.evaluate(self.state)

        engine = self.mcts.engine
        moves = tuple(
            _MoveInfo(
                move_index=move.move_index,
                next_node=self.mcts.nodes[engine.play_move(self.state, move)],
                evaluator_policy=evaluation_result.move_policy[move.move_index])
            for move in engine.playable_moves(self.state)
        )

        self.expanded_info = self.ExpandedInfo(
            total_exploration_count=0,
            result_prediction=evaluation_result.win_rate,
            moves=moves
        )

    def get_move_policy_dict(self) -> dict[int, float]:
        """Returns move policy as a move_index -> policy for playable moves.

        Policy is calculated based on move exploration count and temperature (see MonteCarloTreeSearch.Config).

        This should be used when deciding which move to play, not during simulations.
        """
        assert self.expanded_info, "Node never expanded!"

        def move_policy(move_info: _MoveInfo) -> float:
            assert self.expanded_info
            temperature = self.mcts.config.temperature
            if temperature == 0:
                max_count = max(move_info.exploration_count for move_info in self.expanded_info.moves)
                return 1. if move_info.exploration_count == max_count else 0.
            elif temperature > 0:
                return move_info.exploration_count ** (1 / temperature)
            else:
                raise ValueError(f"Temperature {temperature} is not supported.")

        return {
            move_info.move_index: move_policy(move_info)
            for move_info in self.expanded_info.moves
        }

    def play_move(self) -> _MoveInfo:
        """Selects a move to play.

        The move with the highest Upper Confidence is selected.

        This should be used during simulations, not when deciding for actual move.
        """
        assert self.expanded_info, "Node never expanded!"
        return pick_one_with_highest_value(self.expanded_info.moves,
                                           lambda move_info: self.uct(move_info))

    def uct(self, move_info: _MoveInfo) -> float:
        """Returns Upper Confidence for the given move.

        The Upper Confidence takes into consideration the total exploration count, move exploration count, move reward
        and move policy evaluated by evaluator.
        """
        assert self.expanded_info, "Node never expanded!"

        exploration_rate = self.mcts.config.exploration_rate
        total_exploration_count = self.expanded_info.total_exploration_count

        if total_exploration_count == 0:
            expansion_value = 0.
            exploration_coef = 1.
        else:
            expansion_value = move_info.reward
            exploration_coef = math.sqrt(total_exploration_count) / (1 + move_info.exploration_count)

        return expansion_value + exploration_rate * move_info.evaluator_policy * exploration_coef

    def update(self, move_info: _MoveInfo, result_for_current_player: float) -> None:
        """Updates information for played moves. This should be called during simulations."""
        assert self.expanded_info, "Node never expanded!"
        move_info.update(result_for_current_player)
        self.expanded_info.total_exploration_count += 1

    @property
    def result_prediction(self) -> float:
        """The result predicted by evaluator (or actual result if state represents Game Over state."""
        if self.state.is_game_over:
            assert self.state.result
            return result_for_player(self.state.current_player, self.state.result)
        else:
            assert self.expanded_info, "Node never expanded!"
            return self.expanded_info.result_prediction


class _StateToNodeDefaultDict(dict[State, _Node]):
    """Wrapper around dict[State, _Node] that initializes Node for missing states."""
    mcts: MonteCarloTreeSearch

    def __init__(self, mcts: MonteCarloTreeSearch):
        super().__init__()
        self.mcts = mcts

    def __missing__(self, state: State) -> _Node:
        self[state] = _Node(state, self.mcts)
        return self[state]


@dataclass
class _MoveInfo:
    """The information regarding moves.

    Attributes:
        move_index: The move index.
        next_node: The node achieved by playing this move.
        evaluator_policy: The move policy evaluated by evaluator.
        reward: The reward associated with this move (based on played simulations).
        exploration_count: The number of times this move has been explored.
    """
    move_index: int
    next_node: _Node

    evaluator_policy: float
    reward: float = 0
    exploration_count: int = 0

    def update(self, new_reward: float) -> None:
        """Updates the reward based on the result of the played simulation."""
        self.reward = (self.reward * self.exploration_count + new_reward) / (self.exploration_count + 1)
        self.exploration_count += 1
