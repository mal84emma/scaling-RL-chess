"""ToDo."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ttmpRL import Game, Scorer

import random

import numpy as np

import ttmpRL.model as model
from ttmpRL.scorer import StockfishScorer

from .player import Player


class Agent(Player):
    """AI agent which will play thess.

    Parameters:
        color: chess.WHITE|chess.BLACK. Color the agent is playing is.
        model: Model. Model which provides centipawn score predictions
            of a board position.
    """

    def __init__(self, color, weights_path=None, stockfish_bin=None):
        """ToDo."""
        super().__init__(color)

        assert weights_path is None or stockfish_bin is None, (
            f"You must provide either a model weights path or a Stockfish binary, but not both.\
                You provided: weights_path={weights_path}, stockfish_bin={stockfish_bin}."
        )

        if weights_path is not None:
            self.model: Scorer = model.ChessModel(
                compile_model=True, weights=weights_path
            )
        elif stockfish_bin is not None:  # stockfish for perfect cp scores for testing
            self.model: Scorer = StockfishScorer(stockfish_bin)

    def get_move(self, game: Game) -> str:
        """Searches through all legal moves and returns the move which has
        the lowest predicted score for the opponent (UCI encoded).

        Parameters:
            game: Game. Current game before the move of this agent is made.

        Returns:
            str. UCI encoded movement.
        """
        move = game.NULL_MOVE
        (legal_moves, next_states) = game.get_legal_moves(final_states=True)
        move_scores = []
        for s in next_states:
            move_scores.append(self.model.score_position(s))

        # prevent oscillation by randomly picking from equally good moves
        best_moves = np.argwhere(move_scores == np.amin(move_scores)).flatten().tolist()
        move = random.choice([legal_moves[i] for i in best_moves])

        return move

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
