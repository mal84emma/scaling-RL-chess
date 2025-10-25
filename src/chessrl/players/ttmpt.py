"""Implemention of chess agent using test-time scaling using model predictions,
Test-Time Model Predictive Tuning (TTMPT)."""

__all__ = ("MPAgent",)

import os
import shutil
from timeit import default_timer as timer

import chess
import numpy as np
from tqdm import tqdm

import chessrl.game as game
from chessrl.dataset import PositionDataset
from chessrl.model import train_model
from chessrl.scorer import StockfishScorer
from chessrl.utils import Logger, UCIMove

from .agent import Agent
from .stockfish import Stockfish


class MPAgent(Agent):
    """Chess agent that will use test-time prediction to
    fine-tune the model before making a move."""

    def __init__(
        self,
        color: chess.WHITE | chess.BLACK,
        weights_path: str,
        stockfish_binary: str,
        stockfish_elo: int = 1320,
        n_explore: int = 10,
        ttt_iters: int = 1,
    ):
        super().__init__(color, weights_path)

        self.stockfish_binary = stockfish_binary
        self.stockfish_elo = stockfish_elo
        self.n_explore = n_explore
        self.ttt_iters = ttt_iters

    def get_move(self, board: chess.Board) -> UCIMove:
        """Perform iterations of predicting games and fine-tuning
        the agent before selecting a move."""

        logger = Logger.get_instance()
        timer_start = timer()

        tmp_agent = self._predict_and_tune(board)
        move = tmp_agent.get_move(board)

        timer_end = timer()
        del tmp_agent.model
        del tmp_agent

        logger.info(
            f"TTMP agent made move {move} in {round(timer_end - timer_start, 2)}s.\n"
        )

        return UCIMove(move)

    def _predict_and_tune(self, board: chess.Board) -> Agent:
        """Make predictions of how games will play out from the current
        position and fine-tune the model using score estimates for these
        possible realisations of the game."""

        if len(game.get_legal_moves(board)) == 1:
            # only one move available, no need to tune
            return self.clone()

        logger = Logger.get_instance()

        # setup temporary agents & scorer for predictions
        tmp_agent = self.clone()

        sf_color = chess.BLACK if (self.color == chess.WHITE) else chess.WHITE
        stockfish_agent = Stockfish(
            sf_color,
            self.stockfish_binary,
            self.stockfish_elo,
        )

        stockfish_benchmark = Stockfish(
            self.color,
            self.stockfish_binary,
            elo=3100,  # god tier elo
            stochastic=False,
        )
        scorer = StockfishScorer(self.stockfish_binary)

        # perform prediction-tuning iterations
        logger.info("Making predictions and tuning agent...")
        for _ in tqdm(range(self.ttt_iters), desc="Tuning iterations"):
            next_positions = PositionDataset()

            # play out a few games for each of the estimated `n_explore`
            # top moves, and get score estimates
            # ===
            # get n best moves to explore
            (moves, next_states) = game.get_legal_moves(board, final_states=True)
            score_estimates = [tmp_agent.model.score_position(s) for s in next_states]

            n_best_inds = np.argsort(score_estimates)[: self.n_explore]
            n_best_next_states = [next_states[i] for i in n_best_inds]
            n_best_moves = [moves[i] for i in n_best_inds]

            # playout games for each of the best moves
            for state in tqdm(n_best_next_states, desc="Predicting games", leave=False):
                shat = _playout_and_score(
                    starting_position=state,
                    agent=tmp_agent,
                    stockfish=stockfish_agent,
                    stockfish_scorer=scorer,
                )

                next_positions.add_position(state, shat)
                # end loop

            move_hat = n_best_moves[np.argmin(next_positions.game_scores)]
            logger.info(f"Estimated best move: {move_hat}")
            benchmark_move = stockfish_benchmark.get_move(board)
            # oddly this doesn't always seem to be the best move as suggested by the scorer
            logger.info(f"Benchmark move: {benchmark_move}")
            print("Moves: ", n_best_moves)
            print("Scores:", next_positions.game_scores)

            for _ in range(6):  # copy 2^6 times
                next_positions.append(next_positions)
            next_positions.save("data/positions/tmp_positions.json")

            # retrain the model using new positions data
            # weights are updated inplace
            train_model(
                tmp_agent.model,
                next_positions,
                epochs=5,
                batch_size=len(n_best_inds),
                logdir="data/models/tmp",
            )

            del next_positions
            os.remove("data/positions/tmp_positions.json")
            ## end tuning iteration

        # cleanup
        # TODO: fix memory leak
        shutil.rmtree("data/models/tmp")
        stockfish_agent.close()
        stockfish_benchmark.close()
        scorer.close()

        return tmp_agent


def _playout_and_score(
    starting_position: chess.Board,
    agent: Agent,
    stockfish: Stockfish,
    stockfish_scorer: StockfishScorer,
    n_turns: int = 4,
    n_realisations: int = 1,
) -> float:
    """Play out game n turns against a stockfish agent, starting from
    the current position, for several realisations and report the average
    centipawn score over the trajectories (proxy score for starting position).

    NOTE: agents and engines are passed into this function to avoid init cost."""

    if agent.color == chess.WHITE:
        white_player = agent
        black_player = stockfish
    else:
        white_player = stockfish
        black_player = agent

    mean_scores = []

    # TODO: implement concurrency to speed this up
    for _ in tqdm(range(n_realisations), desc="Playouts", leave=False):
        tmp_board = game.get_board_copy(starting_position)
        score_trajectory = []
        # use mean score over trajectory as models too likely to blunder
        # and destory usefulness of final score

        assert tmp_board.turn == stockfish.color, (
            "Must start out with Stockfish taking a move."
        )
        score_trajectory.append(stockfish_scorer.score_position(tmp_board))

        # play out game a n turns (n moves per player)
        for _ in range(2 * n_turns):
            if game.get_result(tmp_board) is not None:
                break
            else:
                game.next_move(
                    tmp_board,
                    white_player,
                    black_player,
                )
                if tmp_board.turn == stockfish.color:
                    score_trajectory.append(stockfish_scorer.score_position(tmp_board))

        mean_scores.append(np.mean(score_trajectory))

    return np.mean(mean_scores)
