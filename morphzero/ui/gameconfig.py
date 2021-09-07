from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional, Dict

from morphzero.ai.model import Model
from morphzero.core.game import Player, Rules


@unique
class GameType(Enum):
    TIC_TAC_TOE = 1
    GOMOKU = 2
    CONNECT_FOUR = 3


@dataclass(frozen=True)
class PlayerConfig:
    name: str
    ai_model: Optional[Model]


@dataclass(frozen=True)
class GameConfig:
    name: str
    type: GameType
    rules: Rules
    players: Dict[Player, PlayerConfig]

    def __post_init__(self) -> None:
        if self.type == GameType.TIC_TAC_TOE:
            from morphzero.games.genericgomoku.game import GenericGomokuRules
            assert isinstance(self.rules, GenericGomokuRules)
        elif self.type == GameType.GOMOKU:
            from morphzero.games.genericgomoku.game import GenericGomokuRules
            assert isinstance(self.rules, GenericGomokuRules)
        elif self.type == GameType.CONNECT_FOUR:
            from morphzero.games.connectfour.game import ConnectFourRules
            assert isinstance(self.rules, ConnectFourRules)
        else:
            raise ValueError(f"Unsupported Game type: {self.type}")
