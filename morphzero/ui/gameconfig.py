from collections import namedtuple

GameConfig = namedtuple("GameConfig",
                        [
                            "rules",
                            "first_player_name",
                            "first_player_model",
                            "second_player_name",
                            "second_player_model",
                        ])
