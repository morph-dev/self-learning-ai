from typing import Any

from morphzero.ai.algorithms.hash_policy import HashPolicy, StateHashPolicy
from morphzero.ai.algorithms.min_max import MinMaxEvaluator
from morphzero.ai.algorithms.montecarlo import MonteCarloTreeSearch
from morphzero.ai.algorithms.pure_montecarlo import PureMonteCarloTreeSearch
from morphzero.ai.evaluator import EvaluatorModel
from morphzero.ai.trainer import Trainer
from morphzero.common import print_progress_bar, board_to_string
from morphzero.core.common.connect_on_matrix_board import ConnectOnMatrixBoardState
from morphzero.core.game import Rules, Player
from morphzero.games.connectfour.game import ConnectFourRules
from morphzero.games.genericgomoku.game import GenericGomokuRules
from morphzero.ui.gameapp import GameApp
from morphzero.ui.gameconfig import GameType
from morphzero.ui.gameselection import GameConfigParams, PlayerConfigParams, GameSelectionState


def create_game_selection_state() -> GameSelectionState:
    return GameSelectionState([
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
                    config,
                    lambda rules: HashPolicy(
                        rules,
                        StateHashPolicy.load(f"./models/tik_tak_toe_{config}"),
                        HashPolicy.Config.create_for_playing()))
                for config in [
                    "hash_policy_g10000_lr0.2_er0.2",
                    "hash_policy_g100000_lr0.2_er0.3",
                ]
            ] + [
                PlayerConfigParams(
                    "mcts_s1000_exp1.4_temp0.5_t1s_" + config,
                    lambda rules: MonteCarloTreeSearch(
                        rules,
                        HashPolicy(
                            rules,
                            StateHashPolicy.load(f"./models/tik_tak_toe_{config}"),
                            HashPolicy.Config.create_for_playing()),
                        MonteCarloTreeSearch.Config(
                            number_of_simulations=1000,
                            exploration_rate=1.4,
                            temperature=0.5,
                            max_time_sec=1,
                        )
                    )
                )
                for config in [
                    "hash_policy_g10000_lr0.2_er0.2",
                    "hash_policy_g100000_lr0.2_er0.3",
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


def train_hash_policy(rules: Rules,
                      hash_policy_evaluator_config: HashPolicy.Config,
                      trainer_config: Trainer.Config,
                      path: str) -> None:
    model = HashPolicy(rules, StateHashPolicy(), hash_policy_evaluator_config)
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
        hash_policy_evaluator_config=HashPolicy.Config(learning_rate=learning_rate,
                                                       exploration_rate=exploration_rate),
        trainer_config=Trainer.Config(number_of_games=number_of_games),
        path=f"./models/tik_tak_toe_hash_policy" +
             f"_g{number_of_games}_lr{learning_rate}_er{exploration_rate}",
    )


def play() -> None:
    game_selection_state = create_game_selection_state()
    app = GameApp(game_selection_state)
    app.MainLoop()


def pit(game_name: str,
        first_player_name: str,
        second_player_name: str,
        number_of_games: int) -> None:
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
            for s in states:
                assert isinstance(s, ConnectOnMatrixBoardState)
                print()
                print(board_to_string(s.board))
            print(f"Winner is {names[state.result.winner]}")
            break
        else:
            swap_players(names)
            swap_players(model_factory)


if __name__ == "__main__":
    play()
    # train()
    # pit(game_name="Tic Tac Toe",
    #     first_player_name="min-max",
    #     second_player_name="hash_policy_g100000_lr0.2_er0.3",
    #     number_of_games=1000)
