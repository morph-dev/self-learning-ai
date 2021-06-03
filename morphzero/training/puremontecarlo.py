import math
import random
from collections import deque

from morphzero.game.ai_player import AiPlayer
from morphzero.game.base import Player


def _uct(parent, child, exploration_rate=1.4):
    """
    Upper Confidence bounds applied to Trees.
    Returns the score for exploring the child from a given parent node.
    """
    if child.rounds == 0:
        return math.inf
    expansion_value = child.win_ratio
    exploration_value = math.sqrt(math.log(parent.rounds) / child.rounds)
    return expansion_value + exploration_rate * exploration_value


class PureMonteCarloTreeSearch(AiPlayer):
    def __init__(self, rounds, exploration_rate=1.4):
        self.rounds = rounds
        self.exploration_rate = exploration_rate

    def play_move(self, game_engine, state):
        if state.is_game_over:
            return None
        tree = self.build_mcts(game_engine, state)
        win_ratio, move = max([
            (node.win_ratio, move)
            for move, node in tree.children.items()
        ])
        return move

    def build_mcts(self, game_engine, state):
        root = _Node(state)

        for i in range(self.rounds):
            nodes = deque()
            nodes.append(root)
            last_node = root
            # Selection
            while last_node.children is not None:
                last_node = last_node.select_child(self.exploration_rate)
                nodes.append(last_node)
            # Expansion
            if not last_node.state.is_game_over:
                last_node.expand(game_engine)
                last_node = last_node.select_child(self.exploration_rate)
                nodes.append(last_node)
            # Simulation / Playout / Rollout
            state = last_node.state
            while not state.is_game_over:
                move = random.choice(list(game_engine.playable_moves(state)))
                state = game_engine.play_move(state, move)
            # Backpropagation
            while nodes:
                result = state.result.winner
                node = nodes.pop()
                node.update(result)

        return root


class _Node:
    def __init__(self, state):
        self.state = state
        # Wins for self.state.current_player.other_player.
        # It's more useful like that because it's used by parent to decide
        # whether to pick this node or some other sibling.
        self.wins = 0
        self.rounds = 0
        self.children = None

    def expand(self, game_engine):
        if self.state.is_game_over:
            return
        self.children = dict()
        for move in game_engine.playable_moves(self.state):
            self.children[move] = _Node(game_engine.play_move(self.state, move))

    def select_child(self, exploration_rate):
        if self.children is None:
            raise ValueError("Node never expanded!")

        best_children = deque()
        best_value = -math.inf
        for move, child in self.children.items():
            value = _uct(self, child, exploration_rate)
            if math.isclose(value, best_value):
                best_children.append(child)
            elif value > best_value:
                best_children.clear()
                best_children.append(child)
                best_value = value

        return random.choice(best_children)

    def update(self, result):
        self.rounds += 1
        if result == Player.NO_PLAYER:
            self.wins += 0.5
        elif result == self.state.current_player.other_player:
            self.wins += 1

    @property
    def win_ratio(self):
        if self.rounds == 0:
            return 0
        return self.wins / self.rounds

    def __str__(self):
        return f"win_ratio:{self.win_ratio};wins:{self.wins};round:{self.rounds};state:{self.state}"
