
from morphzero.game.genericgomoku import GenericGomokuRules, GenericGomokuGameEngine

def main():
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

if __name__ == "__main__":
    main()
