from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import NamedTuple, Iterable, Optional, Callable, Dict, Deque, Tuple

from morphzero.ai.algorithms.util import pick_one_with_highest_value, result_for_player
from morphzero.ai.base import TrainableModel, EvaluationResult, TrainingData, TrainingSummary, \
    TrainableEvaluator, Evaluator
from morphzero.core.game import State, Result, Rules, Engine


class MonteCarloTreeSearchConfig(NamedTuple):
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


class MonteCarloTreeSearch(TrainableModel, Evaluator):
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
    evaluator: TrainableEvaluator
    config: MonteCarloTreeSearchConfig
    nodes: _StateToNodeDefaultDict

    def __init__(self, rules: Rules, evaluator: TrainableEvaluator, config: MonteCarloTreeSearchConfig):
        if not evaluator.supports_rules(rules):
            raise ValueError("Evaluator doesn't support rules")
        self.rules = rules
        self.engine = self.rules.create_engine()
        self.evaluator = evaluator
        self.config = config

        self.nodes = _StateToNodeDefaultDict(self)

    def supports_rules(self, rules: Rules) -> bool:
        return self.rules == rules

    def evaluate(self, state: State) -> EvaluationResult:
        assert not state.is_game_over, "Can't evaluate state when game is already over"
        start_time_sec = time.time()
        for simulation_index in range(self.config.number_of_simulations):
            elapsed_time_sec = time.time() - start_time_sec
            if self.config.max_time_sec and elapsed_time_sec > self.config.max_time_sec:
                if simulation_index < self.config.number_of_simulations / 2:
                    print(f"Only {simulation_index} out of {self.config.number_of_simulations} simulations.")
                break
            self.simulation(state)

        return self.nodes[state].evaluate()

    def play_move(self, state: State) -> int:
        return self.evaluate(state).pick_best_move()

    def train(self, training_data: TrainingData) -> TrainingSummary:
        return self.evaluator.train(training_data)

    def create_training_data_for_game(
            self, result: Result, states: Iterable[State]) -> TrainingData:
        def get_desired_evaluation_result(state: State) -> EvaluationResult:
            node_evaluation_result = self.nodes[state].evaluate()
            return EvaluationResult(
                win_rate=result_for_player(state.current_player, result),
                move_policy=node_evaluation_result.move_policy,
            )

        return TrainingData(
            tuple(
                (state, get_desired_evaluation_result(state))
                for state in states
                if not state.is_game_over
            )
        )

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
        node_moves = Deque[Tuple[_Node, _MoveInfo]]()
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
            node.state.current_player: node.evaluator_result_prediction,
            node.state.current_player.other_player: 1 - node.evaluator_result_prediction,
        }

        # Backpropagation
        for node, move_info in node_moves:
            node.update(move_info, result_per_player[node.state.current_player])

    @classmethod
    def factory(cls,
                evaluator_factory: Callable[[Rules], TrainableEvaluator],
                config: MonteCarloTreeSearchConfig) -> Callable[[Rules], MonteCarloTreeSearch]:
        return lambda rules: MonteCarloTreeSearch(rules, evaluator_factory(rules), config)


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
    expanded_info: Optional[_ExpandedInfo]

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
                evaluator_policy=evaluation_result.move_policy[move.move_index],
            )
            for move in engine.playable_moves(self.state)
        )

        self.expanded_info = _ExpandedInfo(
            total_exploration_count=0,
            evaluator_result_prediction=evaluation_result.win_rate,
            moves=moves
        )

    def evaluate(self) -> EvaluationResult:
        """Returns EvaluationResult based on exploration count.

        This should be used only after running all simulations (NOT during simulation).
        """
        assert self.expanded_info, "Node never expanded!"
        move_policy = [0.] * self.mcts.rules.number_of_possible_moves()
        for move_info in self.expanded_info.moves:
            move_policy[move_info.move_index] = move_info.exploration_count

        return EvaluationResult.create(
            win_rate=max(move_info.reward for move_info in self.expanded_info.moves),
            move_policy=tuple(move_policy),
            temperature=self.mcts.config.temperature,
            normalize=True,
        )

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
            expansion_value = 0.5  # assume draw
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
    def evaluator_result_prediction(self) -> float:
        """The result predicted by evaluator (or actual result if state represents Game Over state."""
        if self.state.is_game_over:
            assert self.state.result
            return result_for_player(self.state.current_player, self.state.result)
        else:
            assert self.expanded_info, "Node never expanded!"
            return self.expanded_info.evaluator_result_prediction


@dataclass
class _ExpandedInfo:
    """Information available once Node has been expanded.

    Attributes:
        total_exploration_count: The number of times this node has been explored.
        evaluator_result_prediction: The result predicted for the state by the evaluator.
        moves: The tuple of _MoveInfo for playable moves.
    """
    total_exploration_count: int
    evaluator_result_prediction: float
    moves: Tuple[_MoveInfo, ...]


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
    reward: float = 0.5
    exploration_count: int = 0

    def update(self, new_reward: float) -> None:
        """Updates the reward based on the result of the played simulation."""
        self.reward = (self.reward * self.exploration_count + new_reward) / (self.exploration_count + 1)
        self.exploration_count += 1


class _StateToNodeDefaultDict(Dict[State, _Node]):
    """Wrapper around dict[State, _Node] that initializes Node for missing states."""
    mcts: MonteCarloTreeSearch

    def __init__(self, mcts: MonteCarloTreeSearch):
        super().__init__()
        self.mcts = mcts

    def __missing__(self, state: State) -> _Node:
        self[state] = _Node(state, self.mcts)
        return self[state]
