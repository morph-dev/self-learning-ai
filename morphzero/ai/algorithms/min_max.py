from morphzero.ai.algorithms.util import result_for_player
from morphzero.ai.evaluator import Evaluator, EvaluationResult
from morphzero.core.game import State, Rules, Engine, MoveOrMoveIndex


class MinMaxEvaluator(Evaluator):
    """The Evaluator that runs Min-Max algorithm.

    This precisely evaluates the score of the game if both players play optimally, but it has to run over every possible
    state and every possible move from that state. That makes in infeasible for any non trivial game.

    Attributes:
        rules: The rules of the game.
        engine: The game engine.
        score: Maps game State to expected result in range [0,1] from point of view of state.current_player.
    """
    rules: Rules
    engine: Engine
    score: dict[State, float]

    def __init__(self, rules: Rules):
        self.rules = rules
        self.engine = rules.create_engine()
        self.score = dict()

    def supports_rules(self, rules: Rules) -> bool:
        return self.rules == rules

    def evaluate(self, state: State) -> EvaluationResult:
        return EvaluationResult.normalize_and_create(
            self._score_state(state),
            self._get_move_policy(state, 0.01)
        )

    def train(self, learning_data: dict[State, EvaluationResult]) -> None:
        raise TypeError("Training not supported.")

    def _score_state(self, state: State) -> float:
        """Scores the state.

        If state was already scored, it returns result immediately.
        """
        if state in self.score:
            return self.score[state]
        if state.is_game_over:
            assert state.result
            self.score[state] = result_for_player(state.current_player, state.result)
            return self.score[state]

        self.score[state] = max(self._get_move_policy(state))
        return self.score[state]

    def _get_move_policy(self, state: State, min_for_playable_moves: float = 0.) -> tuple[float, ...]:
        """Returns move_policy for a given state.

        Move policy is a tuple of move_scores for each possible move. If move is not playable, it will have 0 value,
        while others will have a value at least min_for_playable_move.

        See _score_move for details on the value of each move.
        """

        def score(move_index: int) -> float:
            if self.engine.is_move_playable(state, move_index):
                move_score = self._score_move(state, move_index)
                return max(move_score, min_for_playable_moves)
            else:
                return 0.

        return tuple(
            score(move_index)
            for move_index in range(self.rules.number_of_possible_moves())
        )

    def _score_move(self, state: State, move: MoveOrMoveIndex) -> float:
        """Returns score of the move for a given state.

        The value is in the range [0, 1] and it's from state.current_player's point of view.
        """
        if not self.engine.is_move_playable(state, move):
            return 0.
        next_state = self.engine.play_move(state, move)
        next_state_score = self._score_state(next_state)
        if state.current_player == next_state.current_player:
            return next_state_score
        elif state.current_player.other_player == next_state.current_player:
            return 1. - next_state_score
        else:
            raise ValueError(f"Unexpected player: {next_state.current_player}")
