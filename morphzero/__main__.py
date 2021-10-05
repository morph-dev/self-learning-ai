from __future__ import annotations

from morphzero.ai.algorithms.hash_policy import HashPolicy
from morphzero.ai.algorithms.montecarlo import MonteCarloTreeSearch, MonteCarloTreeSearchConfig
from morphzero.ai.algorithms.pure_montecarlo import PureMonteCarloTreeSearch, PureMonteCarloTreeSearchConfig
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
                    "pure_mcts_s1000_er1.4_t1s",
                    PureMonteCarloTreeSearch.factory(
                        PureMonteCarloTreeSearchConfig(
                            number_of_simulations=1000,
                            exploration_rate=1.4,
                            max_time_sec=1))),
                PlayerConfigParams(
                    "pure_mcts_s3000_er1.4_t5s",
                    PureMonteCarloTreeSearch.factory(
                        PureMonteCarloTreeSearchConfig(
                            number_of_simulations=3000,
                            exploration_rate=1.4,
                            max_time_sec=5))),
            ] + [
                PlayerConfigParams(
                    config,
                    HashPolicy.factory(f"./models/tic_tac_toe/{config}"))
                for config in [
                    "hash_policy_min_max",
                    "hash_policy__tr_i10000_s1__hash_lr0.3_ex0.2",
                    "hash_policy__tr_i100000_s1__hash_lr0.3_ex0.2",
                    "mcts_hash_policy_g1000_s1000_exp1.4_temp1_lr0.3",
                    "mcts_hash_policy_g10000_s1000_exp1.4_temp1_lr0.3",
                ]
            ] + [
                PlayerConfigParams(
                    "mcts_s1000_exp1.4_temp0.1_t1s_" + config,
                    MonteCarloTreeSearch.factory(
                        HashPolicy.factory(f"./models/tic_tac_toe/{config}"),
                        MonteCarloTreeSearchConfig(
                            number_of_simulations=1000,
                            exploration_rate=1.4,
                            temperature=0.1,
                            max_time_sec=1,
                        )
                    )
                )
                for config in [
                    "hash_policy__tr_i10000_s1__hash_lr0.3_ex0.2",
                    "hash_policy__tr_i100000_s1__hash_lr0.3_ex0.2",
                    "mcts_hash_policy_g1000_s1000_exp1.4_temp1_lr0.3",
                    "mcts_hash_policy_g10000_s1000_exp1.4_temp1_lr0.3",
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
                    PureMonteCarloTreeSearch.factory(
                        PureMonteCarloTreeSearchConfig(
                            number_of_simulations=10000,
                            exploration_rate=1.4,
                            max_time_sec=20))),
            ]
        )
    ])


if __name__ == "__main__":
    game_selection_state = create_game_selection_state()
    app = GameApp(game_selection_state)
    app.MainLoop()
