"""ToDo."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ttmpRL import Game, Scorer

import random

import numpy as np

import ttmpRL.model as model
from ttmpRL.scorer import StockfishScorer
from ttmpRL.utils import get_uci_labels

from .player import Player


class Agent(Player):
    """AI agent which will play thess.

    Parameters:
        model: Model. Model encapsulating the neural network operations.
        move_encodings: list. List of all possible uci movements.
        uci_dict: dict. Dictionary with mappings 'uci'-> int. It's used
        to predict the policy only over the legal movements.
    """

    def __init__(self, color, weights_path=None, stockfish_bin=None):
        """ToDo."""
        super().__init__(color)
        self.uci_dict = {u: i for i, u in enumerate(get_uci_labels())}

        assert weights_path is None or stockfish_bin is None, (
            f"You must provide either a model weights path or a Stockfish binary, but not both.\
                You provided: weights_path={weights_path}, stockfish_bin={stockfish_bin}."
        )

        if weights_path is not None:
            self.model: Scorer = model.ChessModel(
                compile_model=True, weights=weights_path
            )
        elif stockfish_bin is not None:
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
            move_scores.append(self.model.score_position(s)["cp"])

        # move = legal_moves[np.argmin(move_scores)]
        # NOTE: an issue I'm encountering is that when there are multiple
        # 0 score moves, the agent is unable to pick the one that actually wins
        # but Centipawn scores also have their issues, mate is numerically very spikey

        print(move_scores)

        best_moves = np.argwhere(move_scores == np.amin(move_scores)).flatten().tolist()
        move = random.choice([legal_moves[i] for i in best_moves])

        print(move)

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
