from morphzero.ai.algorithms.hash_policy import HashPolicyModel, HashPolicy
from morphzero.ai.algorithms.min_max import MinMaxEvaluator
from morphzero.ai.algorithms.pure_montecarlo import PureMonteCarloTreeSearch
from morphzero.ai.evaluator import EvaluatorModel
from morphzero.ai.trainer import Trainer
from morphzero.core.game import Rules
from morphzero.games.connectfour.game import ConnectFourRules
from morphzero.games.genericgomoku.game import GenericGomokuRules
from morphzero.ui.gameapp import GameApp
from morphzero.ui.gameconfig import GameType
from morphzero.ui.gameselection import GameConfigParams, PlayerConfigParams, GameSelectionState


def train_hash_policy(rules: Rules,
                      hash_policy_evaluator_config: HashPolicyModel.Config,
                      trainer_config: Trainer.Config,
                      path: str) -> None:
    model = HashPolicyModel(rules, HashPolicy(), hash_policy_evaluator_config)
    trainer = Trainer(rules, model, trainer_config)
    trainer.train()
    model.policy.store(path)


def train() -> None:
    # train_tic_tac_toe
    number_of_games = 100000
    learning_rate = 0.2
    exploration_rate = 0.3
    train_hash_policy(
        rules=GenericGomokuRules.create_tic_tac_toe_rules(),
        hash_policy_evaluator_config=HashPolicyModel.Config(learning_rate=learning_rate,
                                                            exploration_rate=exploration_rate),
        trainer_config=Trainer.Config(number_of_games=number_of_games,
                                      print_ratio_increment=0.05),
        path=f"./models/tik_tak_toe_hash_policy" +
             f"_g{number_of_games}_lr{learning_rate}_er{exploration_rate}",
    )


def play() -> None:
    game_selection_state = GameSelectionState([
        GameConfigParams(
            "Tic Tac Toe",
            GameType.TIC_TAC_TOE,
            GenericGomokuRules.create_tic_tac_toe_rules(),
            [
                PlayerConfigParams("Human", None),
                PlayerConfigParams(
                    "min-max",
                    lambda rules: EvaluatorModel.create_with_best_move_picker(MinMaxEvaluator(rules))
                ),
                PlayerConfigParams(
                    "pure_mcts_s1000_er1.4_t1s",
                    lambda rules: EvaluatorModel.create_with_best_move_picker(
                        PureMonteCarloTreeSearch(
                            rules,
                            PureMonteCarloTreeSearch.Config(
                                number_of_simulations=1000,
                                exploration_rate=1.4,
                                max_time_sec=1)))),
                PlayerConfigParams(
                    "pure_mcts_s3000_er1.4_t5s",
                    lambda rules: EvaluatorModel.create_with_best_move_picker(
                        PureMonteCarloTreeSearch(
                            rules,
                            PureMonteCarloTreeSearch.Config(
                                number_of_simulations=3000,
                                exploration_rate=1.4,
                                max_time_sec=5)))),
            ] + [
                PlayerConfigParams(
                    f"hash_policy_{config}",
                    lambda rules: HashPolicyModel(
                        rules,
                        HashPolicy.load(f"./models/tik_tak_toe_hash_policy_{config}"),
                        HashPolicyModel.Config(learning_rate=0, exploration_rate=0)))
                for config in [
                    "g10000_lr0.2_er0.2",
                    "g100000_lr0.2_er0.3",
                ]
            ]
        ),
        GameConfigParams(
            "Gomoku",
            GameType.GOMOKU,
            GenericGomokuRules.create_gomoku_rules(),
            [
                PlayerConfigParams("Human", None),
            ]
        ),
        GameConfigParams(
            "Connect 4",
            GameType.CONNECT_FOUR,
            ConnectFourRules.create_default_rules(),
            [
                PlayerConfigParams("Human", None),
                PlayerConfigParams(
                    "pure_mcts_s10000_er1.4_t20s",
                    lambda rules: EvaluatorModel.create_with_best_move_picker(
                        PureMonteCarloTreeSearch(
                            rules,
                            PureMonteCarloTreeSearch.Config(
                                number_of_simulations=10000,
                                exploration_rate=1.4,
                                max_time_sec=20)))),
            ]
        )
    ])
    app = GameApp(game_selection_state)
    app.MainLoop()


if __name__ == "__main__":
    play()
    # train()
