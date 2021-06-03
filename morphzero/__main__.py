import os.path

from morphzero.game.connectfour import ConnectFourRules
from morphzero.game.genericgomoku import GenericGomokuRules, GenericGomokuGameEngine
from morphzero.training.hashgomoku import HashGomokuTrainer, HashGomokuModel
from morphzero.ui.gameapp import GameApp
from morphzero.ui.gameconfig import GameType
from morphzero.ui.gameselection import GameConfigParams, PlayerConfigParams, GameSelectionState


def train_model(path, iterations, learning_rate, exploration_rate):
    if os.path.isfile(path):
        model = HashGomokuModel.deserialize(path)
    else:
        model = HashGomokuModel()

    rules = GenericGomokuRules(board_size=(3, 3), goal=3)
    engine = GenericGomokuGameEngine(rules)
    trainer = HashGomokuTrainer(engine, learning_rate=learning_rate, exploration_rate=exploration_rate, model=model)

    print_at_ratio = 0.
    for i in range(iterations):
        if round(print_at_ratio * iterations) <= i:
            print(f"Training {round(print_at_ratio * 100)}%")
            print_at_ratio += 0.05
        state = engine.new_game()
        trainer.on_game_start()
        while not state.is_game_over:
            move = trainer.play_move(state)
            state = engine.play_move(state, move)
        trainer.on_game_end(state)

    model.serialize(path)


def main():
    # train_config = {
    #     "iterations": 10000,
    #     "learning_rate": 0.3,
    #     "exploration_rate": 0.2,
    # }
    # path_format = "./models/hash_gomoku_i{iterations}_lr{learning_rate}_er{exploration_rate}.model"
    # trainModel(path_format.format(**train_config), **train_config)

    game_selection_state = GameSelectionState([
        GameConfigParams(
            "Tic Tac Toe",
            GameType.TIC_TAC_TOE,
            GenericGomokuRules.create_tic_tac_toe_rules(),
            [
                PlayerConfigParams("Human", None),
                PlayerConfigParams(
                    "i100000_lr0.1_er0.3",
                    lambda: HashGomokuModel.deserialize("./models/hash_gomoku_i100000_lr0.1_er0.3.model")),
                PlayerConfigParams(
                    "i10000_lr0.3_er0.2",
                    lambda: HashGomokuModel.deserialize("./models/hash_gomoku_i10000_lr0.3_er0.2.model")),
            ]
        ),
        GameConfigParams(
            "Gomoku",
            GameType.GOMOKU,
            GenericGomokuRules.create_gomoku_rules(),
            [PlayerConfigParams("Human", None)]
        ),
        GameConfigParams(
            "Connect 4",
            GameType.CONNECT_FOUR,
            ConnectFourRules.create_default_rules(),
            [PlayerConfigParams("Human", None)]
        )
    ])
    app = GameApp(game_selection_state)
    app.MainLoop()


if __name__ == "__main__":
    main()
