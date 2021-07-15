from collections import defaultdict
from typing import Any, Callable

from morphzero.ai.algorithms.hash_policy import HashPolicy, StateHashPolicy
from morphzero.ai.algorithms.min_max import MinMaxEvaluator
from morphzero.ai.algorithms.montecarlo import MonteCarloTreeSearch
from morphzero.ai.algorithms.pure_montecarlo import PureMonteCarloTreeSearch
from morphzero.ai.evaluator import EvaluatorModel, Evaluator
from morphzero.ai.trainer import Trainer
from morphzero.common import print_progress_bar, board_to_string
from morphzero.core.common.connect_on_matrix_board import ConnectOnMatrixBoardState
from morphzero.core.game import Rules, Player
from morphzero.games.connectfour.game import ConnectFourRules
from morphzero.games.genericgomoku.game import GenericGomokuRules
from morphzero.ui.gameapp import GameApp
from morphzero.ui.gameconfig import GameType
from morphzero.ui.gameselection import GameConfigParams, PlayerConfigParams, GameSelectionState


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


def train() -> None:
    number_of_games = 10000

    hash_policy_learning_rate = 0.2
    hash_policy_exploration_rate = 0.3

    mcts_number_of_simulations = 1000
    mcts_exploration_rate = 1.4
    mcts_temperature = 1

    # train hash_policy
    # hash_policy_learning_rate = 0.2
    # hash_policy_exploration_rate = 0.3
    # train_hash_policy(
    #     rules=GenericGomokuRules.create_tic_tac_toe_rules(),
    #     hash_policy_evaluator_config=HashPolicy.Config(learning_rate=hash_policy_learning_rate,
    #                                                    exploration_rate=hash_policy_exploration_rate),
    #     trainer_config=Trainer.Config(number_of_games),
    #     path=f"./models/tic_tac_toe_hash_policy" +
    #          f"_g{number_of_games}_lr{hash_policy_learning_rate}_er{hash_policy_exploration_rate}",
    # )

    # train tic_tac_toe MCTS with Hash Policy
    train_mcts_with_hash_policy(
        rules=GenericGomokuRules.create_tic_tac_toe_rules(),
        mcts_config=MonteCarloTreeSearch.Config(
            number_of_simulations=mcts_number_of_simulations,
            exploration_rate=mcts_exploration_rate,
            temperature=mcts_temperature,
            max_time_sec=None,
        ),
        hash_policy_config=HashPolicy.Config(hash_policy_learning_rate, exploration_rate=0),
        trainer_config=Trainer.Config(number_of_games),
        path=f"./models/tic_tac_toe_mcts_hash_policy" +
             f"_g{number_of_games}" +
             f"_s{mcts_number_of_simulations}_exp{mcts_exploration_rate}_temp{mcts_temperature}" +
             f"_lr{hash_policy_learning_rate}"
    )


def min_max_to_hash_policy(path: str = "./models/tic_tac_toe_hash_policy_min_max") -> None:
    rules = GenericGomokuRules.create_tic_tac_toe_rules()
    min_max_evaluator = MinMaxEvaluator(rules)
    min_max_evaluator.evaluate(rules.create_engine().new_game())
    hash_policy = StateHashPolicy(min_max_evaluator.score)
    hash_policy.store(path)
    print(len(hash_policy))


def play() -> None:
    game_selection_state = create_game_selection_state()
    app = GameApp(game_selection_state)
    app.MainLoop()


def pit(game_name: str,
        first_player_name: str,
        second_player_name: str,
        number_of_games: int,
        until_first_non_draw: bool) -> None:
    game_selection_state = create_game_selection_state()
    game_config_params = next(
        game_config_params
        for game_config_params in game_selection_state.game_config_params_list
        if game_config_params.name == game_name
    )
    names = {
        Player.FIRST_PLAYER: first_player_name,
        Player.SECOND_PLAYER: second_player_name,
    }
    model_factory = {
        player: next(
            player_config_params.ai_model_factory
            for player_config_params in game_config_params.player_config_params_list
            if player_config_params.default_name == name and player_config_params.ai_model_factory
        )
        for player, name in names.items()
    }

    def swap_players(d: dict[Player, Any]) -> None:
        d[Player.FIRST_PLAYER], d[Player.SECOND_PLAYER] = d[Player.SECOND_PLAYER], d[Player.FIRST_PLAYER]

    rules = game_config_params.rules
    engine = rules.create_engine()
    score_distribution = {
        "draw": 0,
        first_player_name: 0,
        second_player_name: 0,
    }
    for i in range(number_of_games):
        print_progress_bar(i + 1, number_of_games, suffix=f"{i + 1} out of {number_of_games}")
        models = {
            player: factory(rules)
            for player, factory in model_factory.items()
        }

        state = engine.new_game()
        states = [state]
        while not state.is_game_over:
            state = engine.play_move(state, models[state.current_player].play_move(state))
            states.append(state)

        assert state.result
        if state.result.winner != Player.NO_PLAYER:
            score_distribution[names[state.result.winner]] += 1
            if until_first_non_draw:
                for s in states:
                    assert isinstance(s, ConnectOnMatrixBoardState)
                    print()
                    print(board_to_string(s.board))
                break
        else:
            score_distribution["draw"] += 1

        swap_players(names)
        swap_players(model_factory)

    print("Score distribution: ")
    for name, score in score_distribution.items():
        print(f"\t{name}: {score}")


def create_game_selection_state() -> GameSelectionState:
    def evaluator_model_factory(evaluator_factory: Callable[[Rules], Evaluator]) -> Callable[[Rules], EvaluatorModel]:
        return lambda rules: EvaluatorModel.create_with_best_move_picker(evaluator_factory(rules))

    def hash_policy_factory(path: str) -> Callable[[Rules], HashPolicy]:
        return lambda rules: HashPolicy(
            rules,
            StateHashPolicy.load(path),
            HashPolicy.Config.create_for_playing())

    def pure_mcts_factory(config: PureMonteCarloTreeSearch.Config) -> Callable[[Rules], PureMonteCarloTreeSearch]:
        return lambda rules: PureMonteCarloTreeSearch(rules, config)

    def mcts_factory(
            evaluator_factory: Callable[[Rules], Evaluator],
            config: MonteCarloTreeSearch.Config
    ) -> Callable[[Rules], MonteCarloTreeSearch]:
        return lambda rules: MonteCarloTreeSearch(rules, evaluator_factory(rules), config)

    return GameSelectionState([
        GameConfigParams(
            "Tic Tac Toe",
            GameType.TIC_TAC_TOE,
            GenericGomokuRules.create_tic_tac_toe_rules(),
            [
                PlayerConfigParams("Human", None),
                PlayerConfigParams(
                    "pure_mcts_s1000_er1.4_t1s",
                    evaluator_model_factory(
                        pure_mcts_factory(
                            PureMonteCarloTreeSearch.Config(
                                number_of_simulations=1000,
                                exploration_rate=1.4,
                                max_time_sec=1)))),
                PlayerConfigParams(
                    "pure_mcts_s3000_er1.4_t5s",
                    evaluator_model_factory(
                        pure_mcts_factory(
                            PureMonteCarloTreeSearch.Config(
                                number_of_simulations=3000,
                                exploration_rate=1.4,
                                max_time_sec=5)))),
            ] + [
                PlayerConfigParams(
                    config,
                    hash_policy_factory(f"./models/tic_tac_toe_{config}"))
                for config in [
                    "hash_policy_min_max",
                    "hash_policy_g10000_lr0.2_er0.3",
                    "hash_policy_g100000_lr0.2_er0.3",
                    "mcts_hash_policy_g10000_s100_exp1.4_temp1_lr0.2",
                    "mcts_hash_policy_g10000_s1000_exp1.4_temp1_lr0.2",
                ]
            ] + [
                PlayerConfigParams(
                    "mcts_s1000_exp1.4_temp0.1_t1s_" + config,
                    mcts_factory(
                        hash_policy_factory(f"./models/tic_tac_toe_{config}"),
                        MonteCarloTreeSearch.Config(
                            number_of_simulations=1000,
                            exploration_rate=1.4,
                            temperature=0.1,
                            max_time_sec=1,
                        )
                    )
                )
                for config in [
                    "hash_policy_g10000_lr0.2_er0.3",
                    "hash_policy_g100000_lr0.2_er0.3",
                    "mcts_hash_policy_g10000_s100_exp1.4_temp1_lr0.2",
                    "mcts_hash_policy_g10000_s1000_exp1.4_temp1_lr0.2",
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
                    evaluator_model_factory(
                        pure_mcts_factory(
                            PureMonteCarloTreeSearch.Config(
                                number_of_simulations=10000,
                                exploration_rate=1.4,
                                max_time_sec=20)))),
            ]
        )
    ])


if __name__ == "__main__":
    play()
    # min_max_to_hash_policy()
    # train()
    # pit("Tic Tac Toe",
    #     "hash_policy_min_max",
    #     # "hash_policy_g10000_lr0.2_er0.3",
    #     # "mcts_s1000_exp1.4_temp0.1_t1s_" + "hash_policy_g10000_lr0.2_er0.3",
    #     # "mcts_hash_policy_g10000_s1000_exp1.4_temp1_lr0.2",
    #     "mcts_s1000_exp1.4_temp0.1_t1s_" + "mcts_hash_policy_g10000_s1000_exp1.4_temp1_lr0.2",
    #     number_of_games=100,
    #     until_first_non_draw=False)
