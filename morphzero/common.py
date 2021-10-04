from typing import Iterable

from morphzero.core.common.matrix_board import MatrixBoard
from morphzero.core.game import Player, Board


def board_to_string(board: Board,
                    no_player_symbol: str = " ",
                    first_player_symbol: str = "X",
                    second_player_symbol: str = "O",
                    include_index: bool = False) -> str:
    """
    Prints human readable board with '|' and '-' as separators between cells and fills the rest
    using appropriate symbols.
    """
    assert isinstance(board, MatrixBoard)

    player_map = {
        Player.NO_PLAYER: no_player_symbol,
        Player.FIRST_PLAYER: first_player_symbol,
        Player.SECOND_PLAYER: second_player_symbol,
    }
    cell_width = 3
    horizontal_cell_separator = "║"
    vertical_cell_separator = "".center(cell_width, "═")
    cross_separator = "╬"
    row_separator = cross_separator.join([vertical_cell_separator] * board.size.columns)

    def cell_converter(cell_value: Player) -> str:
        return f"{player_map[cell_value]}".center(cell_width)

    def row_converter(row: Iterable[Player]) -> str:
        return horizontal_cell_separator.join(cell_converter(v) for v in row)

    rows_str = [row_converter(row) for row in board.rows]
    matrix_str = [
        rows_str[i // 2] if i % 2 == 0 else row_separator
        for i in range(len(rows_str) * 2 - 1)
    ]

    if include_index:
        column_index_row = " ".join(f"{c}".center(cell_width) for c in range(board.size.columns))
        matrix_str = [column_index_row] + matrix_str
        matrix_row_prefix = [
            f"{(matrix_row_index // 2):{cell_width}}" if matrix_row_index % 2 == 1 else " " * cell_width
            for matrix_row_index in range(len(matrix_str))
        ]
        matrix_str = [
            f"{row_prefix} {row_string}"
            for row_prefix, row_string in zip(matrix_row_prefix, matrix_str)
        ]

    return "\n".join(matrix_str)


def print_progress_bar(
        iteration: int,
        total: int,
        prefix: str = "",
        suffix: str = "",
        decimals: int = 1,
        length: int = 100,
        fill: str = "█",
        empty: str = "-",
        print_end: str = "") -> None:
    """Call in a loop to create terminal progress bar."""
    percent_float = 100 * (iteration / float(total))
    percent_str = f"{percent_float:.{str(decimals)}f}"
    filled_length = int(length * iteration // total)
    bar = (fill * filled_length) + (empty * (length - filled_length))

    # Print New Line on Complete
    if iteration == total:
        print_end += "\n"

    print(f"\r{prefix} |{bar}| {percent_str}% {suffix}", end=print_end)
