import math
import random
from collections import Iterable, Sequence, Callable, deque
from typing import TypeVar

from morphzero.core.game import Result, Player


def result_for_player(player: Player, result: Result) -> float:
    """Returns result value for a given player.

    If player is winner, result is 1. If other player is winner, result is 0.
    If NO_PLAYER is winner (draw), result is 0.5.
    """
    if player not in [Player.FIRST_PLAYER, Player.SECOND_PLAYER]:
        raise ValueError(f"Invalid player: {player}")
    winner = result.winner
    if winner == Player.NO_PLAYER:
        return 0.5
    elif winner == player:
        return 1.
    elif winner == player.other_player:
        return 0.
    else:
        raise ValueError(f"Invalid winner: {winner}")


T = TypeVar("T")


def pick_one_index_with_highest_value(items: Sequence[float]) -> int:
    """Returns the index (one of) for the highest value."""
    return pick_one_with_highest_value(range(len(items)), lambda i: items[i])


def pick_one_with_highest_value(items: Iterable[T], key: Callable[[T], float]) -> T:
    """Returns one of the items (if multiple) for which key function returns the highest value."""
    max_key = -math.inf
    max_items = deque[T]()

    for item in items:
        key_value = key(item)
        if key_value > max_key:
            max_key = key_value
            max_items.clear()
        if math.isclose(max_key, key_value):
            max_items.append(item)

    if not max_items:
        raise ValueError("No items found.")

    return random.choice(max_items)
