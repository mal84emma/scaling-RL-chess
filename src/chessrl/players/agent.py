"""ToDo."""

__all__ = ("Agent",)

import random

import chess
import numpy as np

import chessrl
import chessrl.game as game
import chessrl.model as model
from chessrl.scorer import Scorer, StockfishScorer
from chessrl.utils import UCIMove


class Agent:
    """AI agent which will play thess.

    Parameters:
        color: chess.WHITE|chess.BLACK. Color the agent is playing is.
        model: Model. Model which provides centipawn score predictions
            of a board position.
    """

    def __init__(
        self,
        color: chess.WHITE | chess.BLACK,
        weights_path=None,
        stockfish_binary=None,
    ):
        self.color = color

        assert weights_path is None or stockfish_binary is None, (
            f"You must provide either a model weights path or a Stockfish binary, but not both.\
                You provided: weights_path={weights_path}, stockfish_binary={stockfish_binary}."
        )

        if weights_path is not None:
            self.model: Scorer = model.ChessScoreModel(
                compile_model=True,
                weights=weights_path,
            )
        elif stockfish_binary is not None:
            # stockfish for perfect cp scores for testing
            self.model: Scorer = StockfishScorer(stockfish_binary)

    def get_move(self, board: chess.Board) -> UCIMove:
        """Searches through all legal moves and returns the move which has
        the lowest predicted score for the opponent (UCI encoded).

        Parameters:
            board: Current board position for the game before the move of
                this agent is made.

        Returns:
            UCIMove. UCI encoded movement.
        """
        assert board.turn == self.color, "It is not this agent's turn to play."

        move = chessrl.NULL_MOVE
        (legal_moves, next_states) = game.get_legal_moves(board, final_states=True)
        move_scores = []
        for s in next_states:
            move_scores.append(self.model.score_position(s))

        # prevent oscillation by randomly picking from equally good moves
        best_moves = np.argwhere(move_scores == np.amin(move_scores)).flatten().tolist()
        move = random.choice([legal_moves[i] for i in best_moves])

        # TODO: try using WDL scores with mate picker logic

        return UCIMove(move)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

    def clone(self):
        return Agent(self.color, self.model.weights_path)

    def close(self):
        if isinstance(self.model, StockfishScorer):
            self.model.close()
        else:
            del self.model
