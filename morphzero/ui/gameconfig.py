from collections import namedtuple

PlayerConfig = namedtuple("PlayerConfig",
                          ["name", "model"])

GameConfig = namedtuple("GameConfig",
                        ["rules", "players"])
