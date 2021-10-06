import math

from tensorflow import keras

from morphzero.ai.algorithms.hash_policy import HashPolicy
from morphzero.ai.base import Evaluator
from morphzero.common import board_to_string
from morphzero.games.genericgomoku.ai.tic_tac_toe import TicTacToeKeras, TicTacToeKerasConfig
from morphzero.games.genericgomoku.game import GenericGomokuRules


def load_keras_evaluator(path: str) -> Evaluator:
    evaluator = TicTacToeKeras(
        TicTacToeKerasConfig(
            training=True,
            verbose=1,
            learning_rate=0.001,
            epochs=500,
            filters=10,
            mid_layer_size=100,
        )
    )
    evaluator.model = keras.models.load_model(path)
    return evaluator


def compare(min_max: HashPolicy, evaluator: Evaluator) -> None:
    for state in min_max.policy:
        if state.is_game_over:
            continue
        min_max_evaluation_result = min_max.evaluate(state)
        evaluator_evaluation_result = evaluator.evaluate(state)

        win_rate_difference = math.fabs(min_max_evaluation_result.win_rate - evaluator_evaluation_result.win_rate)

        evaluator_best_move_policy = max(evaluator_evaluation_result.move_policy)
        evaluator_best_move = next(
            i
            for i, policy in enumerate(evaluator_evaluation_result.move_policy)
            if policy >= evaluator_best_move_policy
        )
        min_max_best_move_policy = max(min_max_evaluation_result.move_policy)
        found_best_move = min_max_evaluation_result.move_policy[evaluator_best_move] >= min_max_best_move_policy

        if (win_rate_difference > 0.25) or not found_best_move:
            print()
            print(state.current_player)
            print(board_to_string(state.board))
            print("min_max")
            print(min_max_evaluation_result)
            print("evaluator")
            print(evaluator_evaluation_result)
            # print("WINRATE:")
            # print(f"min_max: {min_max_evaluation_result.win_rate:.4f}")
            # print(f"evaluator: {evaluator_evaluation_result.win_rate:.4f}")
            # print("MOVE POLICY")
            # print(" ".join(f"{p:.5f}" for p in min_max_evaluation_result.move_policy))
            # print(" ".join(f"{p:.5f}" for p in evaluator_evaluation_result.move_policy))


def load_and_compare() -> None:
    rules = GenericGomokuRules.create_tic_tac_toe_rules()
    min_max = HashPolicy.factory("./../models/tic_tac_toe/hash_policy_min_max")(rules)
    evaluator = load_keras_evaluator("./../models/tic_tac_toe/based_on_min_max")
    compare(min_max, evaluator)


if __name__ == "__main__":
    load_and_compare()
