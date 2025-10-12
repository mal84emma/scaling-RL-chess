__all__ = ("Stockfish",)

import logging
import os
import random

import chess
import chess.engine
from chess.engine import Limit, SimpleEngine

from chessrl.utils import UCIMove

# Remove anoying warnings of the engine.
chess.engine.LOGGER.setLevel(logging.ERROR)


def _clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class Stockfish:
    """AI using Stockfish to play a game of chess."""

    def __init__(
        self,
        color: chess.WHITE | chess.BLACK,
        binary_path: str | os.PathLike,
        thinking_time=0.1,
        search_depth=10,
        elo=1320,
    ):
        self.color = color

        self.engine: SimpleEngine = SimpleEngine.popen_uci(binary_path)
        # print(list(self.engine.options))
        self.limit: Limit = Limit(time=thinking_time, depth=search_depth)
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        # page below discusses effect of Stockfish level on Elo rating
        # https://chess.stackexchange.com/questions/29860/is-there-a-list-of-approximate-elo-ratings-for-each-stockfish-level

        self.elo = elo

    def get_move(self, board: chess.Board) -> UCIMove:
        assert board.turn == self.color, "It's not Stockfish's turn to play."

        result = self.engine.play(board, self.limit)
        move = result.move.uci()

        # add a bit of stochasticity to Stockfish's move choice, select move
        # either one better or one worse than given elo choice randomly
        variations = self.engine.analyse(board, self.limit, multipv=50)
        move_variations = [v["pv"][0].uci() for v in variations]

        if move in move_variations:
            move_index = move_variations.index(move)
            if random.random() <= 0.5:
                increment = random.choice([-1, 1])
                move_index = _clamp(move_index + increment, 0, len(move_variations) - 1)
                move = move_variations[move_index]

        return UCIMove(move)

    def close(self):
        """Close the connection to the engine."""
        self.engine.quit()
