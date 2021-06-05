import math
import random
import time
from collections import deque

from morphzero.game.ai_player import AiPlayer
from morphzero.game.base import Player


class HeuristicsMonteCarloTreeSearch(AiPlayer):
    def __init__(self, heuristics, rounds, rollout_steps, exploration_rate=1.4, max_time_sec=1):
        self.heuristics = heuristics
        self.rounds = rounds
        self.rollout_steps = rollout_steps
        self.exploration_rate = exploration_rate
        self.max_time_sec = max_time_sec

    def play_move(self, game_engine, state):
        if state.is_game_over:
            raise ValueError("Can't play a move when game is already over")
        tree = self.build_mcts(game_engine, state)
        win_ratio, move = max([
            (1 - node.win_ratio, move)
            for move, node in tree.children.items()
        ])

        print("\n\nEvaluating:")
        for move2, node in tree.children.items():
            print(move2)
            print(node)
        print()
        rows, columns = game_engine.rules.board_size
        # GOMOKU
        from morphzero.game.genericgomoku import GenericGomokuGameEngine
        if isinstance(game_engine, GenericGomokuGameEngine):
            for row in range(rows):
                for column in range(columns):
                    if state.board[row, column] == Player.NO_PLAYER:
                        from morphzero.game.genericgomoku import GenericGomokuMove
                        print(f"{(1 - tree.children[GenericGomokuMove(row, column)].win_ratio):.2f}", end=" ")
                    else:
                        print("_" * 4, end=" ")
                print()
        # CONNECT FOUR
        from morphzero.game.connectfour import ConnectFourGameEngine
        if isinstance(game_engine, ConnectFourGameEngine):
            for column in range(columns):
                if state.board[0, column] == Player.NO_PLAYER:
                    from morphzero.game.connectfour import ConnectFourMove
                    print(f"{(1 - tree.children[ConnectFourMove(column)].win_ratio):.2f}", end=" ")
                else:
                    print("_" * 4, end=" ")
            print()

        print(sorted([
            ((1 - node.win_ratio, ), *move2)
            for move2, node in tree.children.items()
        ], reverse=True))
        return move

    def build_mcts(self, game_engine, state):
        root = _Node(state, prior=0.5)
        start_time_sec = time.time()
        for i in range(self.rounds):
            if self.max_time_sec > 0:
                elapsed_time_sec = time.time() - start_time_sec
                if elapsed_time_sec > self.max_time_sec:
                    print(f"Only {i} out of {self.rounds}")
                    return root

            nodes = deque()
            nodes.append(root)
            last_node = root
            # Selection
            while last_node.children is not None:
                last_node = last_node.select_child(self.exploration_rate)
                nodes.append(last_node)
            # Expansion
            if not last_node.state.is_game_over:
                last_node.expand(game_engine, self.heuristics)
                last_node = last_node.select_child(self.exploration_rate)
                nodes.append(last_node)
            # Simulation / Playout / Rollout
            state = last_node.state
            for step in range(self.rollout_steps):
                if state.is_game_over:
                    break
                potential_move_states = [
                    (move, game_engine.play_move(state, move))
                    for move in game_engine.playable_moves(state)
                ]
                potential_score_move_states = [
                    (
                        self.heuristics.estimate_win_rate_for_current_player(
                            game_engine.rules, state),
                        move,
                        state,
                    )
                    for move, state in potential_move_states
                ]
                max_score = max(score for score, _move, _state in potential_score_move_states)
                best_next_states = [
                    state
                    for score, _move, state in potential_score_move_states
                    if math.isclose(score, max_score)
                ]
                state = random.choice(best_next_states)
            # Backpropagation
            while nodes:
                node = nodes.pop()
                node.rounds += 1
                if state.is_game_over:
                    winner = state.result.winner
                    if winner == node.state.current_player:
                        win_rate = 1
                    elif winner == Player.NO_PLAYER:
                        win_rate = 0.5
                    elif winner == node.state.current_player.other_player:
                        win_rate = 0
                    else:
                        raise ValueError("Unexpected winner")
                    node.wins += win_rate
                else:
                    estimate_win_rate = self.heuristics.estimate_win_rate_for_current_player(
                        game_engine.rules, state)
                    node.wins += estimate_win_rate

        return root


class _Node:
    def __init__(self, state, prior):
        self.state = state
        # Used as a replacement for win_rate when there are no rounds played.
        self.prior = prior
        # Wins for self.state.current_player.
        self.wins = 0
        self.rounds = 0
        # children are None if Node was not expanded.
        # If node is expanded (but not in the process of being expanded), self.rounds will be more than 0.
        self.children = None

    def expand(self, game_engine, heuristics):
        if self.state.is_game_over:
            return
        self.children = dict()
        for move in game_engine.playable_moves(self.state):
            child_state = game_engine.play_move(self.state, move)
            self.children[move] = _Node(
                child_state,
                heuristics.estimate_win_rate_for_current_player(
                    game_engine.rules, child_state))

    def select_child(self, exploration_rate):
        if self.children is None:
            raise ValueError("Node never expanded!")

        best_children = deque()
        best_value = -math.inf
        for move, child in self.children.items():
            value = self.uct(child, exploration_rate)
            if math.isclose(value, best_value):
                best_children.append(child)
            elif value > best_value:
                best_children.clear()
                best_children.append(child)
                best_value = value

        return random.choice(best_children)

    @property
    def win_ratio(self):
        if self.rounds == 0:
            return self.prior
        return self.wins / self.rounds

    def uct(self, child, exploration_rate):
        """
        Upper Confidence bounds applied to Trees.
        Returns the score for exploring the child from a given parent node.
        """
        expansion_value = 1 - child.win_ratio
        exploration_value = math.sqrt(math.log(max(1, self.rounds)) / max(1, child.rounds))
        return expansion_value + exploration_rate * exploration_value

    def __str__(self):
        return "\n\t".join([
            f"win_ratio:{self.win_ratio}",
            f"wins:{self.wins}",
            f"round:{self.rounds}",
            f"prior:{self.prior}",
        ])
