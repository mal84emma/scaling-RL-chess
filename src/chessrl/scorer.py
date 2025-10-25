import logging
from typing import Protocol

import chess
from chess.engine import Limit, SimpleEngine

# Remove annoying warnings of the engine.
chess.engine.LOGGER.setLevel(logging.ERROR)

import chessrl


class Scorer(Protocol):
    def score_position(self, board: chess.Board) -> int | dict:
        """Evaluates the strength of a board position for the player that is
        about to take a turn."""
        ...

    def close(self) -> None:
        """Close connections to resources."""
        ...


class StockfishScorer:
    """A chess position scorer that uses the Stockfish engine to evaluate positions.

    This class uses a Stockfish chess engine instance to calculate board position
    scores for the player that is about to take a turn.
    """

    def __init__(
        self,
        binary_path: str,
        thinking_time: float = 0.01,
        search_depth: int = 10,
    ):
        """Initialize the ScorerStockfish with a Stockfish engine.

        Args:
            binary_path: Path to the Stockfish binary executable
            thinking_time: Time limit for engine analysis in seconds (default: 0.01)
            search_depth: Maximum search depth for engine analysis (default: 10)
        """

        self.engine: SimpleEngine = SimpleEngine.popen_uci(binary_path)
        self.limit: Limit = Limit(time=thinking_time, depth=search_depth)

    def score_position(self, board: chess.Board, cp_only: bool = True) -> dict:
        """Evaluates the strength of a board position for the player that is
        about to take a turn.

        Args:
            board: chess.Board. The board position to evaluate.
            cp_only: bool, Whether to return only `cp_score` as float for easier
                interfacing.

        Returns:
            scores: dict, A dictionary of scores which has the following key value
                   pairs; {'cp': cp_score, 'rate': score_rate}

        The scoring provides two different evaluations:
        - cp_score: Centipawn evaluation from the perspective of the current player
        - score_rate: Win/draw/loss score expectation from the perspective of the current player

        References:
        - https://stackoverflow.com/questions/69861415/how-to-get-the-winning-chances-of-white-in-python-chess
        - https://python-chess.readthedocs.io/en/latest/engine.html#chess.engine.Score
        """

        scores = self.engine.analyse(board, self.limit)["score"]
        cp_score = scores.pov(board.turn).score(mate_score=chessrl.MATE_CP_SCORE)
        score_rate = scores.wdl().pov(board.turn).expectation()

        if cp_only:
            return cp_score

        return {"cp": cp_score, "rate": score_rate}

    def close(self):
        """Close the connection to the engine."""
        self.engine.quit()
