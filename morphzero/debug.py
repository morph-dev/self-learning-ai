from morphzero.ai.algorithms.hash_policy import HashPolicy, HashPolicyConfig
from morphzero.common import board_to_string
from morphzero.games.genericgomoku.game import GenericGomokuRules


def hash_policy_debug() -> None:
    rules = GenericGomokuRules.create_tic_tac_toe_rules()
    engine = rules.create_engine()
    model = HashPolicy.factory(
        f"./../models/tic_tac_toe/hash_policy__tr_i10000_s1__hash_lr0.3_ex0.2",
        HashPolicyConfig(learning_rate=0, exploration_rate=0, temperature=1.)
    )(rules)

    state = engine.new_game()
    for move in [3, 6, 4, 5, 0, ]:
        state = engine.play_move(state, move)
    print(board_to_string(state.board))
    print(model.evaluate(state))


if __name__ == "__main__":
    hash_policy_debug()
