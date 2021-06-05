from collections import defaultdict

from morphzero.common import Directions, BoardCoordinates, is_inside_matrix
from morphzero.game.base import Player


class Heuristics:
    def estimate_win_rate_for_current_player(self, rules, state):
        """
        Returns value in [0, 1] range.
        """
        raise NotImplementedError


class LineConnectionHeuristics(Heuristics):
    """
    The simple heuristics that estimates the score based on how many open lines with potentials each player has.
    """

    def __init__(self, backoff, start_score, current_player_boost=1.2):
        self.backoff = backoff
        self.start_score = start_score
        self.current_player_boost = current_player_boost

    def score(self, connected, goal):
        index = goal - connected
        if index < len(self.backoff):
            return self.backoff[index]
        else:
            return 0

    def estimate_win_rate_for_current_player(self, rules, state):
        # set = rules.goal connected cells which has only one player in it

        # ← ↖ ↑ ↗
        directions = [-direction for direction in Directions.HALF_INTERCARDINAL]

        # for each coordinate
        #   for each set in direction ← ↖ ↑ ↗
        #     for each player
        #       how many there are in the set cells
        board_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # for each player
        #   for each length
        #     how many sets there are
        player_connection_counts = defaultdict(lambda: [0] * (rules.goal + 1))

        rows, columns = rules.board_size
        board = state.board
        for row in range(rows):
            for column in range(columns):
                coordinates = BoardCoordinates(row, column)
                for direction in directions:
                    board_counts[coordinates][direction] = board_counts[coordinates + direction][direction].copy()
                    board_counts[coordinates][direction][board[coordinates]] += 1

                    far_outside_coordinates = coordinates + (direction * rules.goal)
                    if is_inside_matrix(far_outside_coordinates, rules.board_size):
                        board_counts[coordinates][direction][board[far_outside_coordinates]] -= 1

                    far_inside_coordinates = far_outside_coordinates - direction
                    if is_inside_matrix(far_inside_coordinates, rules.board_size):
                        for player in [Player.FIRST_PLAYER, Player.SECOND_PLAYER]:
                            if board_counts[coordinates][direction][player.other_player] == 0:
                                player_connection_counts[player][board_counts[coordinates][direction][player]] += 1

        player_score = {
            player: sum(
                self.score(connected, rules.goal) * player_connection_counts[player][connected]
                for connected in range(rules.goal + 1)
            )
            for player in [Player.FIRST_PLAYER, Player.SECOND_PLAYER]
        }

        current_player_score = self.start_score + player_score[state.current_player]
        current_player_score *= self.current_player_boost
        other_player_score = self.start_score + player_score[state.current_player.other_player]

        if current_player_score + other_player_score == 0:
            score = 0
        else:
            score = (current_player_score - other_player_score) / (current_player_score + other_player_score)

        # score is at range [-1, 1]
        # we should scale it at [0, 1]
        score = (score + 1) / 2
        return score
