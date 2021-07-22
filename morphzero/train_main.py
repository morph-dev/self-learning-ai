from morphzero.ai.algorithms.hash_policy import HashPolicy, StateHashPolicy
from morphzero.ai.algorithms.montecarlo import MonteCarloTreeSearch
from morphzero.ai.evaluator import EvaluationResult
from morphzero.ai.trainer import Trainer
from morphzero.common import board_to_string
from morphzero.core.game import Rules
from morphzero.games.genericgomoku.ai.tic_tac_toe import TicTacToeKeras
from morphzero.games.genericgomoku.game import GenericGomokuRules


def train_hash_policy(
        rules: Rules,
        hash_policy_evaluator_config: HashPolicy.Config,
        trainer_config: Trainer.Config,
        path: str) -> None:
    model = HashPolicy(rules, StateHashPolicy(), hash_policy_evaluator_config)
    trainer = Trainer(rules, model, trainer_config)
    trainer.train()
    model.policy.store(path)


def train_mcts_with_hash_policy(
        rules: Rules,
        mcts_config: MonteCarloTreeSearch.Config,
        hash_policy_config: HashPolicy.Config,
        trainer_config: Trainer.Config,
        path: str) -> None:
    hash_policy = HashPolicy(rules, StateHashPolicy(), hash_policy_config)
    model = MonteCarloTreeSearch(
        rules,
        hash_policy,
        mcts_config
    )
    trainer = Trainer(rules, model, trainer_config)
    trainer.train()
    hash_policy.policy.store(path)


def train_keras():
    rules = GenericGomokuRules.create_tic_tac_toe_rules()
    ttt = TicTacToeKeras(
        TicTacToeKeras.Config(
            training=True,
            verbose=0,
            learning_rate=0.1,
            epochs=10,
            filters=10,
            mid_layer_size=100,
        )
    )
    mcts = MonteCarloTreeSearch(
        rules,
        ttt,
        MonteCarloTreeSearch.Config(1000))
    trainer = Trainer(rules,
                      mcts,
                      Trainer.Config(100))

    min_max = HashPolicy.factory("./../models/tic_tac_toe/hash_policy_min_max")(rules)
    engine = rules.create_engine()
    state = engine.new_game()
    state = engine.play_move(state, 4)

    def print_result(result: EvaluationResult) -> None:
        print(result.win_rate)
        print(" ".join([f"{mv:.5f}" for mv in result.move_policy]))

    r1 = ttt.evaluate(state)
    trainer.train()
    r2 = ttt.evaluate(state)

    print(board_to_string(state.board))
    print_result(min_max.evaluate(state))
    print_result(r1)
    print_result(r2)


def train() -> None:
    number_of_games = 10000

    hash_policy_learning_rate = 0.3
    hash_policy_exploration_rate = 0.2

    mcts_number_of_simulations = 1000
    mcts_exploration_rate = 1.4
    mcts_temperature = 1

    # # train hash_policy
    # train_hash_policy(
    #     rules=GenericGomokuRules.create_tic_tac_toe_rules(),
    #     hash_policy_evaluator_config=HashPolicy.Config(learning_rate=hash_policy_learning_rate,
    #                                                    exploration_rate=hash_policy_exploration_rate),
    #     trainer_config=Trainer.Config(number_of_games),
    #     path=f"./../models/tic_tac_toe/hash_policy" +
    #          f"_g{number_of_games}_lr{hash_policy_learning_rate}_er{hash_policy_exploration_rate}",
    # )

    # # train tic_tac_toe MCTS with Hash Policy
    # train_mcts_with_hash_policy(
    #     rules=GenericGomokuRules.create_tic_tac_toe_rules(),
    #     mcts_config=MonteCarloTreeSearch.Config(
    #         number_of_simulations=mcts_number_of_simulations,
    #         exploration_rate=mcts_exploration_rate,
    #         temperature=mcts_temperature,
    #         max_time_sec=None,
    #     ),
    #     hash_policy_config=HashPolicy.Config(hash_policy_learning_rate, exploration_rate=0),
    #     trainer_config=Trainer.Config(number_of_games),
    #     path=f"./../models/tic_tac_toe/mcts_hash_policy" +
    #          f"_g{number_of_games}" +
    #          f"_s{mcts_number_of_simulations}_exp{mcts_exploration_rate}_temp{mcts_temperature}" +
    #          f"_lr{hash_policy_learning_rate}"
    # )

    train_keras()


if __name__ == "__main__":
    train()
