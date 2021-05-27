from game.base import Player


def get_result_message(state, player_names):
    result = state.result
    if result is None:
        raise ValueError("Game is not over yet!")
    elif result == Player.NO_PLAYER:
        return "It is a draw!"
    else:
        return f"Winner is {player_names[result].name}!"
