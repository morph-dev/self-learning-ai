from __future__ import annotations

from typing import Union, Iterator

import numpy as np

from morphzero.core.common.connect_on_matrix_board import ConnectOnMatrixBoardResult, ConnectOnMatrixBoardState, \
    ConnectOnMatrixBoardRules, ConnectOnMatrixBoardEngine, ConnectOnMatrixBoardMove
from morphzero.core.common.matrix_board import MatrixBoardSize, MatrixBoardCoordinates
from morphzero.core.game import Player


class ConnectFourResult(ConnectOnMatrixBoardResult):
    """Result for the Connect 4 game."""
    pass


class ConnectFourState(ConnectOnMatrixBoardState):
    """State for the Connect 4 game."""
    pass


class ConnectFourMove(ConnectOnMatrixBoardMove):
    """Move for the Connect 4 game."""
    pass


MoveOrMoveIndex = Union[ConnectFourMove, int]


class ConnectFourRules(ConnectOnMatrixBoardRules):
    def number_of_possible_moves(self) -> int:
        return ConnectFourEngine.number_of_possible_moves(self.board_size)

    def create_engine(self) -> ConnectFourEngine:
        return ConnectFourEngine(self)

    @classmethod
    def create_default_rules(cls) -> ConnectFourRules:
        return cls(board_size=MatrixBoardSize(6, 7), goal=4)


class ConnectFourEngine(ConnectOnMatrixBoardEngine):

    def new_game(self) -> ConnectFourState:
        return ConnectFourState(
            current_player=Player.FIRST_PLAYER,
            result=None,
            board=np.full(self.rules.board_size, Player.NO_PLAYER))

    def create_move_from_move_index(self, move_index: int) -> ConnectFourMove:
        return ConnectFourMove(
            move_index=move_index,
            resign=move_index == self.get_move_index_for_resign(),
            coordinates=self.get_coordinates_for_move_index(move_index))

    def playable_moves(self, state: ConnectFourState) -> Iterator[ConnectFourMove]:  # type: ignore[override]
        if state.is_game_over:
            return []
        for column in range(self.rules.board_size.columns):
            if state.board[0][column] == Player.NO_PLAYER:
                for row in range(self.rules.board_size.rows):
                    # We found move if we are on the last row or next row is not EMPTY.
                    if row == self.rules.board_size.rows - 1 or state.board[row + 1][column] != Player.NO_PLAYER:
                        coordinates = MatrixBoardCoordinates(row, column)
                        yield ConnectFourMove(
                            move_index=self.get_move_index_for_coordinates(coordinates),
                            resign=False,
                            coordinates=coordinates)
                        break
        yield ConnectFourMove(
            move_index=self.get_move_index_for_resign(),
            resign=True,
            coordinates=None)

    def playable_moves_bitmap(self, state: ConnectFourState) -> tuple[bool, ...]:  # type: ignore[override]
        result = [False] * self.number_of_possible_moves(self.rules.board_size)
        for move in self.playable_moves(state):
            result[move.move_index] = True
        return tuple(result)

    def is_move_playable(  # type: ignore[override]
            self, state: ConnectFourState, move: MoveOrMoveIndex) -> bool:
        if state.is_game_over:
            return False
        if isinstance(move, int):
            move = self.create_move_from_move_index(move)
        else:
            self.validate_move(move)
        if move.resign:
            return True
        assert move.coordinates
        if state.board[move.coordinates] == Player.NO_PLAYER:
            next_row = move.coordinates.row + 1
            if next_row == self.rules.board_size.rows:
                # last row
                return True
            new_row_value: Player = state.board[next_row][move.coordinates.column]
            return Player.NO_PLAYER != new_row_value
        return False

    def play_move(  # type: ignore[override]
            self, state: ConnectFourState, move: MoveOrMoveIndex) -> ConnectFourState:
        if isinstance(move, int):
            move = self.create_move_from_move_index(move)
        if not self.is_move_playable(state, move):
            raise ValueError(f"Move {move} is not playable.")

        board = state.board.copy()
        if move.resign:
            return ConnectFourState(
                current_player=state.current_player.other_player,
                result=ConnectFourResult.create_resignation(
                    winner=state.current_player.other_player),
                board=board)
        assert move.coordinates

        board[move.coordinates] = state.current_player
        result = ConnectFourResult.create_from_board_and_last_move(self.rules, board, move.coordinates)
        if result:
            # game over
            return ConnectFourState(
                current_player=state.current_player.other_player,
                result=result,
                board=board)
        else:
            return ConnectFourState(
                current_player=state.current_player.other_player,
                result=None,
                board=board)
