import pickle
from typing import List

from morphzero.ai.algorithms.hash_policy import HashPolicy, HashPolicyConfig
from morphzero.ai.base import EvaluationResult, TrainingData
from morphzero.common import board_to_string
from morphzero.games.connectfour.game import ConnectFourRules, ConnectFourState
from morphzero.games.genericgomoku.game import GenericGomokuRules


def debug_tic_tac_toe_hash_policy() -> None:
    rules = GenericGomokuRules.create_tic_tac_toe_rules()
    engine = rules.create_engine()
    model = HashPolicy.factory(
        "./../models/tic_tac_toe/hash_policy__tr_i10000_s1__hash_lr0.3_ex0.2",
        HashPolicyConfig(learning_rate=0, exploration_rate=0, temperature=1.)
    )(rules)

    state = engine.new_game()
    print(board_to_string(state.board))
    print(model.evaluate(state))

    for move in [3, 6, 4, 5, 0, ]:
        state = engine.play_move(state, move)
        print(board_to_string(state.board))
        print(model.evaluate(state))


def debug_connect4_hash_policy() -> None:
    rules = ConnectFourRules.create_default_rules()
    engine = rules.create_engine()
    model = HashPolicy.factory(
        "./../models/connect4/hash_policy_mcts__tr_i10_s1__hash_lr0.3_ex0__mcts_sim1000_ex1.4_temp1",
        HashPolicyConfig(learning_rate=0, exploration_rate=0, temperature=1.)
    )(rules)

    def evaluate(state: ConnectFourState, print_result: bool = False) -> EvaluationResult:
        evaluation_result = model.evaluate(state)
        if print_result:
            print(evaluation_result.win_rate)
            moves = [
                engine.playable_move_for_column(state, column)
                for column in range(rules.board_size.columns)
            ]
            move_policy = [
                evaluation_result.move_policy[move.move_index] if move else 0.
                for move in moves
            ]
            print(", ".join(f"{policy:.4f}" for policy in move_policy))
        return evaluation_result

    state = engine.new_game()
    print(board_to_string(state.board))
    evaluate(state, print_result=True)

    for move_column in [3, 2]:
        move = engine.playable_move_for_column(state, move_column)
        assert move
        state = engine.play_move(state, move)
        print(board_to_string(state.board))
        evaluate(state, print_result=True)


def debug_connect4_training_data() -> None:
    with open(
            "./../models/connect4/hash_policy_mcts__tr_i10_s1__hash_lr0.3_ex0__mcts_sim1000_ex1.4_temp1_training_data",
            "rb") as f:
        all_training_data: List[TrainingData] = pickle.load(f)
        for training_data in all_training_data:
            print()
            print("NEW TRAINING DATA")
            for state, evaluation_result in training_data.data:
                print(board_to_string(state.board))
                print(evaluation_result.win_rate)


if __name__ == "__main__":
    # debug_tic_tac_toe_hash_policy()
    debug_connect4_hash_policy()
    # debug_connect4_training_data()
