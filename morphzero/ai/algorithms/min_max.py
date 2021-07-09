from typing import Iterable

from morphzero.ai.algorithms.util import result_for_player
from morphzero.ai.evaluator import Evaluator, EvaluationResult
from morphzero.core.game import Result, State, Rules, Engine, MoveOrMoveIndex


class MinMaxEvaluator(Evaluator):
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
        return EvaluationResult(
            self._score_state(state),
            self._get_move_policy(state, 0.01)
        )

    def train(self, result: Result, states: Iterable[State]) -> None:
        raise TypeError("Training not supported.")

    def _score_state(self, state: State) -> float:
        if state in self.score:
            return self.score[state]
        if state.is_game_over:
            assert state.result
            self.score[state] = result_for_player(state.current_player, state.result)
            return self.score[state]

        self.score[state] = max(self._get_move_policy(state))
        return self.score[state]

    def _get_move_policy(self, state: State, min_for_playable_moves: float = 0) -> tuple[float, ...]:
        return tuple(
            self._score_move(state, move_index, min_for_playable_moves)
            for move_index in range(self.rules.number_of_possible_moves())
        )

    def _score_move(self, state: State, move: MoveOrMoveIndex, min_for_playable_moves: float) -> float:
        if not self.engine.is_move_playable(state, move):
            return 0
        next_state = self.engine.play_move(state, move)
        next_state_score = self._score_state(next_state)
        if state.current_player == next_state.current_player:
            return max(next_state_score, min_for_playable_moves)
        elif state.current_player.other_player == next_state.current_player:
            return max(1 - next_state_score, min_for_playable_moves)
        else:
            raise ValueError(f"Unexpected player: {next_state.current_player}")
