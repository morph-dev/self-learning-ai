from __future__ import annotations

from typing import Union, Iterator

import numpy as np

from morphzero.core.common.connect_on_matrix_board import ConnectOnMatrixBoardResult, ConnectOnMatrixBoardState, \
    ConnectOnMatrixBoardRules, ConnectOnMatrixBoardEngine, ConnectOnMatrixBoardMove
from morphzero.core.common.matrix_board import MatrixBoardSize, MatrixBoardCoordinates
from morphzero.core.game import Player


class GenericGomokuResult(ConnectOnMatrixBoardResult):
    """Result for the Gomoku type games."""


class GenericGomokuState(ConnectOnMatrixBoardState):
    """State for the Gomoku type games."""


class GenericGomokuMove(ConnectOnMatrixBoardMove):
    """Move for the Gomoku type games."""


MoveOrMoveIndex = Union[GenericGomokuMove, int]


class GenericGomokuRules(ConnectOnMatrixBoardRules):
    def number_of_possible_moves(self) -> int:
        return GenericGomokuEngine.number_of_possible_moves(self.board_size)

    def create_engine(self) -> GenericGomokuEngine:
        return GenericGomokuEngine(self)

    @classmethod
    def create_tic_tac_toe_rules(cls) -> GenericGomokuRules:
        return cls(board_size=MatrixBoardSize(3, 3), goal=3)

    @classmethod
    def create_gomoku_rules(cls) -> GenericGomokuRules:
        return cls(board_size=MatrixBoardSize(15, 15), goal=5)


class GenericGomokuEngine(ConnectOnMatrixBoardEngine):

    def new_game(self) -> GenericGomokuState:
        return GenericGomokuState(
            current_player=Player.FIRST_PLAYER,
            result=None,
            board=np.full(self.rules.board_size, Player.NO_PLAYER))

    def create_move_from_move_index(self, move_index: int) -> GenericGomokuMove:
        return GenericGomokuMove(
            move_index=move_index,
            resign=move_index == self.get_move_index_for_resign(),
            coordinates=self.get_coordinates_for_move_index(move_index))

    def playable_moves(self, state: GenericGomokuState) -> Iterator[GenericGomokuMove]:  # type: ignore[override]
        if state.is_game_over:
            return []
        for row in range(self.rules.board_size.rows):
            for column in range(self.rules.board_size.columns):
                if state.board[row, column] == Player.NO_PLAYER:
                    coordinates = MatrixBoardCoordinates(row, column)
                    yield GenericGomokuMove(
                        move_index=self.get_move_index_for_coordinates(coordinates),
                        resign=False,
                        coordinates=coordinates)
        yield GenericGomokuMove(
            move_index=self.get_move_index_for_resign(),
            resign=True,
            coordinates=None)

    def playable_moves_bitmap(self, state: GenericGomokuState) -> tuple[bool, ...]:  # type: ignore[override]
        result = [False] * self.number_of_possible_moves(self.rules.board_size)
        for move in self.playable_moves(state):
            result[move.move_index] = True
        return tuple(result)

    def is_move_playable(  # type: ignore[override]
            self, state: GenericGomokuState, move: MoveOrMoveIndex) -> bool:
        if state.is_game_over:
            return False
        if isinstance(move, int):
            move = self.create_move_from_move_index(move)
        else:
            self.validate_move(move)
        coordinates = move.coordinates
        if coordinates is None:
            # resign move
            return True
        board_player: Player = state.board[move.coordinates]
        return board_player == Player.NO_PLAYER

    def play_move(  # type: ignore[override]
            self, state: GenericGomokuState, move: MoveOrMoveIndex) -> GenericGomokuState:
        if isinstance(move, int):
            move = self.create_move_from_move_index(move)
        if not self.is_move_playable(state, move):
            raise ValueError(f"Move {move} is not playable.")
        board = state.board.copy()
        if move.resign:
            return GenericGomokuState(
                current_player=state.current_player.other_player,
                result=GenericGomokuResult.create_resignation(
                    winner=state.current_player.other_player),
                board=board)
        assert move.coordinates
        board[move.coordinates] = state.current_player
        result = GenericGomokuResult.create_from_board_and_last_move(self.rules, board, move.coordinates)
        if result:
            # game over
            return GenericGomokuState(
                current_player=state.current_player.other_player,
                result=result,
                board=board)
        else:
            return GenericGomokuState(
                current_player=state.current_player.other_player,
                result=None,
                board=board)