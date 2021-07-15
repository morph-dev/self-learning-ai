from typing import Any

from morphzero.ai.algorithms.hash_policy import HashPolicy
from morphzero.ai.algorithms.montecarlo import MonteCarloTreeSearch
from morphzero.common import print_progress_bar, board_to_string
from morphzero.core.common.connect_on_matrix_board import ConnectOnMatrixBoardState
from morphzero.core.game import Player, Rules
from morphzero.games.genericgomoku.game import GenericGomokuRules
from morphzero.ui.gameselection import PlayerConfigParams


def battle(
        rules: Rules,
        players: dict[Player, PlayerConfigParams],
        number_of_games: int,
        until_first_non_draw: bool) -> None:
    names = {
        player: players[player].default_name
        for player in players
    }
    model_factory = {
        player: players[player].ai_model_factory
        for player in players
    }

    def swap_players(d: dict[Player, Any]) -> None:
        d[Player.FIRST_PLAYER], d[Player.SECOND_PLAYER] = d[Player.SECOND_PLAYER], d[Player.FIRST_PLAYER]

    engine = rules.create_engine()
    score_distribution = {
        "draw": 0,
        names[Player.FIRST_PLAYER]: 0,
        names[Player.SECOND_PLAYER]: 0,
    }
    for i in range(number_of_games):
        print_progress_bar(i + 1, number_of_games, suffix=f"{i + 1} out of {number_of_games}")
        models = {
            player: factory(rules)
            for player, factory in model_factory.items()
            if factory
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


if __name__ == "__main__":
    battle(
        GenericGomokuRules.create_tic_tac_toe_rules(),
        {
            Player.FIRST_PLAYER: PlayerConfigParams(
                "min-max",
                HashPolicy.factory("./../models/tic_tac_toe/hash_policy_min_max")),
            Player.SECOND_PLAYER: PlayerConfigParams(
                "mcts_s1000_exp1.4_temp0.1_t1s_",
                MonteCarloTreeSearch.factory(
                    HashPolicy.factory("./../models/tic_tac_toe/mcts_hash_policy_g10000_s1000_exp1.4_temp1_lr0.3"),
                    MonteCarloTreeSearch.Config(
                        number_of_simulations=1000,
                        exploration_rate=1.4,
                        temperature=0.1,
                        max_time_sec=1))),
        },
        number_of_games=100,
        until_first_non_draw=False,
    )
