"""Implemention of chess agent using test-time scaling using model predictions,
Test-Time Model Predictive Tuning (TTMPT)."""

__all__ = ("TTAgent",)

import traceback
from timeit import default_timer as timer

import chess
import numpy as np
from tqdm import tqdm

import chessrl.game as game
from chessrl.dataset import GameDataset
from chessrl.utils import Logger, UCIMove

from .agent import Agent


# TODO: this needs serious reworking to implement the new tuning strategy
def _playout_and_save_game(
    dataset: GameDataset,
    starting_game,
    agent: Agent,
    stockfish_binary: str,
) -> GameDataset:
    """Get agent to play out game against stockfish, starting from particular
    position, and add to dataset."""

    logger = Logger.get_instance()

    tmp_board = game.get_board_copy(starting_game)
    ...
    white_player = agent
    black_player = ...

    # play game
    try:
        while game.get_result(tmp_board) is None:
            agent_move = agent.get_move(tmp_board, real_game=True)
            game.move(tmp_board, agent_move)

    except Exception:
        logger.error(traceback.format_exc())

    dataset.append(tmp_board)

    white_player.close()
    black_player.close()
    # will also need to close the scorer object (if stockfish is used there)

    return dataset


class TTAgent(Agent):
    """Chess agent that will use test-time prediction to
    fine-tune the model before making a move."""

    def __init__(
        self,
        color,
        weights_path=None,
        ttsf_bin="../../res/stockfish-17-macos-m1-apple-silicon",
        tt_games=32,
        ttt_iters=3,
    ):
        super().__init__(color, weights_path)

        self.ttsf_bin = ttsf_bin
        self.tt_games = tt_games
        self.ttt_iters = ttt_iters

    def get_move(self, board: chess.Board) -> UCIMove:
        """Perform iterations of predicting games and fine-tuning
        the agent before selecting a move."""

        logger = Logger.get_instance()
        timer_start = timer()

        move = game.NULL_MOVE

        tmp_agent = self.predict_and_tune(board)
        policy = tmp_agent.predict_policy(board)
        move = game.get_legal_moves()[np.argmax(policy)]

        timer_end = timer()
        del tmp_agent
        logger.info(
            f"### TTMP agent made move {move} in {round(timer_end - timer_start, 2)}s.\n"
        )

        return UCIMove(move)

    def predict_and_tune(self, board: chess.Board) -> Agent:
        logger = Logger.get_instance()

        tmp_agent = self.clone()

        logger.info("Making predictions and tuning agent...")
        for r in tqdm(range(self.ttt_iters), desc="Tuning iterations"):
            dataset = GameDataset()

            # play lots of games from the curent board state onwards
            for _ in tqdm(range(self.tt_games), desc="Predicting games"):
                dataset = _playout_and_save_game(
                    dataset=dataset,
                    starting_game=board,
                    agent=tmp_agent,
                    stockfish_binary=self.ttsf_bin,
                )
            dataset.save("../../data/tmp.json")

            # retrain the model using new data
            tmp_agent.train(
                dataset,
                epochs=3,
                validation_split=0.0,
                batch_size=8,
                logdir="../../data/models/tmp",
            )

        return tmp_agent
