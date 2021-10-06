from typing import Any, Dict, Tuple

from morphzero.ai.algorithms.hash_policy import HashPolicy, HashPolicyConfig
from morphzero.common import print_progress_bar, board_to_string
from morphzero.core.common.connect_on_matrix_board import ConnectOnMatrixBoardState
from morphzero.core.game import Player, Rules
from morphzero.games.genericgomoku.game import GenericGomokuRules
from morphzero.ui.gameselection import PlayerConfigParams


def battle(
        rules: Rules,
        players: Tuple[PlayerConfigParams, ...],
        number_of_games: int,
        until_first_non_draw: bool) -> None:
    players_dict = {
        Player.FIRST_PLAYER: players[0],
        Player.SECOND_PLAYER: players[1],
    }

    def swap_players(d: Dict[Player, Any]) -> None:
        d[Player.FIRST_PLAYER], d[Player.SECOND_PLAYER] = d[Player.SECOND_PLAYER], d[Player.FIRST_PLAYER]

    engine = rules.create_engine()
    score_distribution = {
        "draw": 0,
        players_dict[Player.FIRST_PLAYER].default_name: 0,
        players_dict[Player.SECOND_PLAYER].default_name: 0,
    }
    print("p1:", players_dict[Player.FIRST_PLAYER].default_name)
    print("p2:", players_dict[Player.SECOND_PLAYER].default_name)
    for i in range(number_of_games):
        print_progress_bar(
            i + 1,
            number_of_games,
            suffix=f"- {i + 1} / {number_of_games}\t" +
                   ", ".join((
                       f"draw: {score_distribution['draw']}",
                       f"p1: {score_distribution[players_dict[Player.FIRST_PLAYER].default_name]}",
                       f"p2: {score_distribution[players_dict[Player.SECOND_PLAYER].default_name]}",
                   )),
        )
        models = {
            player: player_config.ai_model_factory(rules)
            for player, player_config in players_dict.items()
            if player_config.ai_model_factory
        }

        state = engine.new_game()
        states = [state]
        while not state.is_game_over:
            state = engine.play_move(state, models[state.current_player].play_move(state))
            states.append(state)

        assert state.result
        if state.result.winner != Player.NO_PLAYER:
            score_distribution[players_dict[state.result.winner].default_name] += 1
            if until_first_non_draw:
                for s in states:
                    assert isinstance(s, ConnectOnMatrixBoardState)
                    print()
                    print(board_to_string(s.board))
                break
        else:
            score_distribution["draw"] += 1

        swap_players(players_dict)

    print("Score distribution: ")
    for name, score in score_distribution.items():
        print(f"\t{name}: {score}")


if __name__ == "__main__":
    battle(
        GenericGomokuRules.create_tic_tac_toe_rules(),
        tuple(
            PlayerConfigParams(
                filename,
                HashPolicy.factory(
                    f"./../models/tic_tac_toe/{filename}",
                    HashPolicyConfig(learning_rate=0., exploration_rate=0., temperature=0.1),
                )
            )
            for filename in [
                # "hash_policy_min_max",
                "hash_policy__tr_i10000_s1__hash_lr0.3_ex0.2",
                "hash_policy__tr_i100000_s1__hash_lr0.3_ex0.2",
            ]
        ),
        number_of_games=100,
        until_first_non_draw=False,
    )
