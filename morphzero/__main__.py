import os.path

from morphzero.game.genericgomoku import GenericGomokuRules, GenericGomokuGameEngine
from morphzero.training.hashgomoku import HashGomokuTrainer, HashGomokuModel
from ui.gameconfigapp import RulesConfig, GameConfigApp, ModelConfig
from ui.genericgomokuui import GenericGomokuApp


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

    rules_config = [
        RulesConfig(
            "TicTacToe",
            GenericGomokuRules(board_size=(3, 3), goal=3),
            [
                ModelConfig("Human", None, None),
                ModelConfig("i100000_lr0.1_er0.3",
                            "./models/hash_gomoku_i100000_lr0.1_er0.3.model",
                            HashGomokuModel.deserialize),
                ModelConfig("i10000_lr0.3_er0.2",
                            "./models/hash_gomoku_i10000_lr0.3_er0.2.model",
                            HashGomokuModel.deserialize),
            ]
        ),
        RulesConfig(
            "Gomoku (15x15)",
            GenericGomokuRules(board_size=(15, 15), goal=5),
            [ModelConfig("Human", None, None)]
        ),
        RulesConfig(
            "Gomoku (19x19)",
            GenericGomokuRules(board_size=(19, 19), goal=5),
            [ModelConfig("Human", None, None)]
        ),
    ]

    app = GameConfigApp(rules_config)
    app.mainloop()

    game_config = app.game_config
    if game_config is None:
        return
    print(game_config)

    app = GenericGomokuApp(game_config)
    app.mainloop()


if __name__ == "__main__":
    main()
