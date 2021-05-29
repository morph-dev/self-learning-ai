from game.base import Player


def get_result_message(winner, player_names):
    if winner is None:
        raise ValueError("Game is not over yet!")
    elif winner == Player.NO_PLAYER:
        return "It is a draw!"
    else:
        return f"Winner is {player_names[winner].name}!"
