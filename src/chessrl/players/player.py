__all__ = ("Player",)

from typing import Protocol

import chess

from chessrl.utils import UCIMove


class Player(Protocol):
    color: bool

    def get_move(self, board: chess.Board) -> UCIMove:
        """Gets the move the player wants to make for the given
        game position in UCI notation (e.g. f2f3 or g8f6).
        """
        ...
