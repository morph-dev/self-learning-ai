from collections import namedtuple
from enum import Enum, unique


@unique
class GameType(Enum):
    TIC_TAC_TOE = 1
    GOMOKU = 2
    CONNECT_FOUR = 3


GameConfig = namedtuple("GameConfig",
                        ["name", "type", "rules", "players"])

PlayerConfig = namedtuple("PlayerConfig",
                          ["name", "ai_player"])
