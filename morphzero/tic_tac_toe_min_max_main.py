from morphzero.ai.algorithms.hash_policy import StateHashPolicy
from morphzero.ai.algorithms.min_max import MinMaxEvaluator
from morphzero.games.genericgomoku.game import GenericGomokuRules


def min_max_to_hash_policy(path: str = "./models/tic_tac_toe_hash_policy_min_max") -> None:
    rules = GenericGomokuRules.create_tic_tac_toe_rules()
    min_max_evaluator = MinMaxEvaluator(rules)
    min_max_evaluator.evaluate(rules.create_engine().new_game())
    hash_policy = StateHashPolicy(min_max_evaluator.score)
    hash_policy.store(path)
    print(f"Done! Stored {len(hash_policy)} states")


if __name__ == "__main__":
    min_max_to_hash_policy("./../models/tic_tac_toe/hash_policy_min_max")
