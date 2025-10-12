from __future__ import annotations

__all__ = ("Player",)

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from chessrl import Game


class Player(Protocol):
    color: bool

    def get_move(self, game: Game) -> str:
        """Gets the move the player wants to make for the given
        game position in UCI notation (e.g. f2f3 or g8f6).
        """
        ...
