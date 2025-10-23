"""Benchmark a trained agent by playing agaisnt Stockfish.
Plays several games and reports a summary of the results.
"""

import argparse
import os
import random
import traceback
from timeit import default_timer as timer

import chess
import matplotlib.pyplot as plt
from tqdm import tqdm

from chessrl import Agent, Logger, MPAgent, Stockfish, StockfishScorer, game
from chessrl.model import get_model_path
from chessrl.utils import init_board_image, update_board_image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def play_game(
    model_path: str,
    stockfish_binary: str,
    use_opening_book: bool = True,
    stockfish_elo: int = 1320,
    log: bool = False,
    use_ttmp: bool = False,
    plot: bool = False,
    delay: float = 0.5,
):
    """Plays a game and returns the result..

    Parameters:
        model_path: path to the .weights.h5 model for agent to use.
        stockfish_binary: str. Path to stockfish binary.
        stockfish_elo: int. Stockfish difficulty.
    """

    logger = Logger.get_instance()

    board = game.get_new_board()

    if use_opening_book:
        game.set_opening_position(board, book="resources/opening_book.json")

    agent_color = chess.WHITE if random.random() <= 0.5 else chess.BLACK
    stockfish_color = chess.BLACK if agent_color is chess.WHITE else chess.WHITE

    stockfish = Stockfish(stockfish_color, stockfish_binary, elo=stockfish_elo)

    # set up agent
    agent_type = MPAgent if use_ttmp else Agent
    chess_agent = agent_type(color=agent_color, weights_path=model_path)
    # chess_agent = Stockfish(agent_color, stockfish_binary, elo=2500)

    if agent_color is chess.WHITE:
        white_player = chess_agent
        black_player = stockfish
    else:
        white_player = stockfish
        black_player = chess_agent

    if log:
        agent_color_name = chess.COLOR_NAMES[agent_color]
        logger.info(f"Starting game: Agent is {agent_color_name}")

    if plot:
        scorer = StockfishScorer(binary_path=stockfish_binary)
        im, ax, fig = init_board_image(board, agent_color, delay)
        update_board_image(board, agent_color, im, ax, fig, scorer, delay)

    try:
        timer_start = timer()

        # play out game
        while game.get_result(board) is None:
            # make move
            game.next_move(board, white_player, black_player)

            if plot:
                update_board_image(board, agent_color, im, ax, fig, scorer, delay)

        timer_end = timer()

        if log:
            logger.info(
                f"Game done. Result: {game.get_result(board)}. "
                f"took {round(timer_end - timer_start, 2)} secs"
            )
    except Exception:
        logger.error(traceback.format_exc())

    res = {"color": agent_color, "result": game.get_result(board)}

    # close out engine instances
    chess_agent.close()
    stockfish.close()
    if plot:
        scorer.close()
        plt.close(fig)

    return res


def benchmark(
    model_dir: str,
    stockfish_binary: str,
    games: int = 10,
    use_opening_book: bool = True,
    stockfish_elo: int = 1320,
    log: bool = False,
    plot: bool = False,
    delay: float = 0.5,
    use_ttmp: bool = False,
):
    """Plays N games and gets stats about the results.

    Parameters:
        model_dir: str. Directory where the neural net weights and training
            logs will be saved.
        stockfish_binary: str. Path to stockfish binary.
    """

    if log:
        logger = Logger.get_instance()

    model_path = get_model_path(model_dir)

    results = [
        play_game(
            model_path,
            stockfish_binary,
            use_opening_book=use_opening_book,
            stockfish_elo=stockfish_elo,
            log=log,
            use_ttmp=use_ttmp,
            plot=plot,
            delay=delay,
        )
        for _ in tqdm(range(games), desc="Games played")
    ]

    if log:
        logger.debug("Calculating stats.")

    color_ints = [1 if x["color"] == chess.WHITE else -1 for x in results]
    winners = [x["result"] for x in results]  # 0 for draw
    won = [a * b for a, b in zip(color_ints, winners)]

    wins = len([x for x in won if x == 1])
    draws = len([x for x in results if x["result"] == 0])

    if log:
        print("##################### SUMMARY ###################")
        print(f"Games played: {games}")
        print(f"Games won: {wins}")
        print(f"Games drawn: {draws}")
        print("#################################################")

    return dict(
        played=games,
        won=wins,
        drawn=draws,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plays some chess games againstStockfish and reports the results."
    )
    parser.add_argument(
        "model_dir",
        metavar="model dir",
        help="Directory containing model weights file.",
    )
    parser.add_argument(
        "stockfish_binary",
        metavar="stockfish binary",
        help="Stockfish binary path.",
    )
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--delay", metavar="delay", type=float, default=0.5)

    args = parser.parse_args()

    benchmark(
        args.model_dir,
        args.stockfish_binary,
        log=True,
        plot=args.plot,
        delay=args.delay,
        use_ttmp=False,
        games=10,
        stockfish_elo=1320,
    )
    # see page 77 of https://web.ist.utl.pt/diogo.ferreira/papers/ferreira13impact.pdf
    # for comparison between stockfish depth and Elo rating
    # note: I've put in random stockfish depth just to make the games different
