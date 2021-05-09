from morphzero.game.genericgomoku import GenericGomokuRules, GenericGomokuGameEngine
from morphzero.training.hashgomoku import HashGomokuTrainer, HashGomokuModel
from morphzero.ui.genericgomokuui import GenericGomokuApp

import random
import os.path

def playHumanVsHuman():
    rules = GenericGomokuRules(board_size=(3,3), goal=3, first_player_name="Iks", second_player_name="Oks")
    engine = GenericGomokuGameEngine(rules)
    state = engine.new_game()
    print(state)
    while not state.is_game_over:
        while True:
            move_str = input("Enter move to play: ")
            move = tuple(int(x) for x in move_str.split())
            if engine.is_move_playable(state, move):
                break
            print(f"Move {move} is not playable move ({list(engine.playable_moves(state))}).")
        state = engine.play_move(state, move)
        print(state)
    print(f"Game is over! Result: {state.result}")

def playHumanVsAi(path, debug):
    rules = GenericGomokuRules(board_size=(3,3), goal=3, first_player_name="Iks", second_player_name="Oks")
    engine = GenericGomokuGameEngine(rules)
    model = HashGomokuModel.deserialize(path)

    def play_human(state):
        print(state)
        while True:
            move_str = input("Enter move to play: ")
            move = tuple(int(x) for x in move_str.split())
            if engine.is_move_playable(state, move):
                return move
            print(f"Move {move} is not playable move ({list(engine.playable_moves(state))}).")

    def play_ai(state):
        print(state)
        if debug:
            for move in engine.playable_moves(state):
                print(move, model.state_policy.get(engine.play_move(state, move), None))
        move = model.play_move(engine, state)
        print(f"Ai plays {move}")
        return move

    play_as = int(input("Play as:\n1. Iks\n2. Oks\n"))
    if play_as == 1:
        current_player = play_human
        other_player = play_ai
    elif play_as == 2:
        current_player = play_ai
        other_player = play_human
    else:
        print("Unknown input")
        return

    state = engine.new_game()
    while not state.is_game_over:
        move = current_player(state)
        state = engine.play_move(state, move)
        current_player, other_player = other_player, current_player

    print(state)
    print(f"Game is over! Result: {state.result}")

def trainModel(path, iterations, learning_rate, exploration_rate):
    if os.path.isfile(path):
        model = HashGomokuModel.deserialize(path)
    else:
        model = HashGomokuModel()

    rules = GenericGomokuRules(board_size=(3,3), goal=3, first_player_name="Iks", second_player_name="Oks")
    engine = GenericGomokuGameEngine(rules)
    trainer = HashGomokuTrainer(engine, learning_rate=learning_rate, exploration_rate=exploration_rate, model = model)

    print_at_ratio = 0.
    for i in range(iterations):
        if round(print_at_ratio * iterations) <= i:
            print(f"Training {round(print_at_ratio*100)}%")
            print_at_ratio += 0.05
        state = engine.new_game()
        trainer.on_game_start()
        while not state.is_game_over:
            move = trainer.play_move(state)
            state = engine.play_move(state, move)
        trainer.on_game_end(state)

    model.serialize(path)
def gomoku_ui():
    rules = GenericGomokuRules(board_size=(3,3), goal=3, first_player_name="Iks", second_player_name="Oks")
    # rules = GenericGomokuRules(board_size=(15,15), goal=5, first_player_name="Iks", second_player_name="Oks")
    app = GenericGomokuApp(rules)
    app.mainloop()

def main():
    # playHumanVsHuman()

    # train_config = {
    #     "iterations": 10000,
    #     "learning_rate": 0.3,
    #     "exploration_rate": 0.2,
    # }
    # path_format = "./models/hash_gomoku_i{iterations}_lr{learning_rate}_er{exploration_rate}.model"
    # trainModel(path_format.format(**train_config), **train_config)

    # playHumanVsAi(
    #     # "./models/hash_gomoku_i100000_lr0.1_er0.3.model",
    #     "./models/hash_gomoku_i10000_lr0.3_er0.2.model",
    #     # debug=True,
    # )

    gomoku_ui()

if __name__ == "__main__":
    main()
