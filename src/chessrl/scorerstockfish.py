import logging
import chess.engine
from chess.engine import SimpleEngine, Limit
from scorer import Scorer
from game import Game

# Remove annoying warnings of the engine.
chess.engine.LOGGER.setLevel(logging.ERROR)


class ScorerStockfish(Scorer):
    """A chess position scorer that uses the Stockfish engine to evaluate positions.
    
    This class implements the Scorer interface and uses a Stockfish chess engine
    to calculate board position scores for the player that is about to take a turn.
    """

    def __init__(self, binary_path: str, thinking_time: float = 0.01, search_depth: int = 10):
        """Initialize the ScorerStockfish with a Stockfish engine.

        Args:
            binary_path: Path to the Stockfish binary executable
            thinking_time: Time limit for engine analysis in seconds (default: 0.01)
            search_depth: Maximum search depth for engine analysis (default: 10)
        """

        self.engine: SimpleEngine = SimpleEngine.popen_uci(binary_path)
        self.limit: Limit = Limit(time=thinking_time, depth=search_depth)

    def score_position(self, game: Game) -> dict:
        """Evaluates the strength of a board position for the player that is
        about to take a turn.

        Args:
            game: An object of the Game class which describes a game position
            
        Returns:
            scores: A dictionary of scores which has the following key value 
                   pairs; {'cp': cp_score, 'rate': score_rate}

        The scoring provides two different evaluations:
        - cp_score: Centipawn evaluation from the perspective of the current player
        - score_rate: Win/draw/loss score expectation from the perspective of the current player

        References:
        - https://stackoverflow.com/questions/69861415/how-to-get-the-winning-chances-of-white-in-python-chess
        - https://python-chess.readthedocs.io/en/latest/engine.html#chess.engine.Score
        """

        scores = self.engine.analyse(game.board, self.limit)['score']
        cp_score = scores.pov(game.board.turn).score()
        score_rate = scores.wdl().pov(game.board.turn).expectation()

        return {'cp': cp_score, 'rate': score_rate}

    def close(self):
        """Close the engine connection to free resources."""
        if hasattr(self, 'engine') and self.engine:
            self.engine.quit()
    
    def __del__(self):
        """Destructor to ensure engine is properly closed."""
        self.close()
