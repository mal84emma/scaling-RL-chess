"""Generate training games by playing two Stockfish models against each other."""

import argparse
import random
from concurrent.futures import ThreadPoolExecutor

import chess
from tqdm import tqdm

import chessrl
import chessrl.game as game
import chessrl.utils as utils
from chessrl import Agent, GameDataset, Stockfish
from chessrl.utils import Logger


def play_game(
    stockfish_binary: str,
    dataset: GameDataset,
    tqbar: tqdm = None,
):
    """Play a game between two Stockfish instances with randomised ELOs
    and add to dataset."""

    # randomly samples ELOs for intermediate to advanced players as suggested by
    # https://www.chess.com/blog/bomb2030/what-is-considered-a-good-elo-score-for-recreational-players-is-1400-good
    # use cross-correlation to make games competitive
    white_elo = round(random.randint(1400, 2000), -1)
    black_elo = white_elo + round(random.randint(-300, 300), -1)
    black_elo = utils.clamp(black_elo, 1400, 2000)

    white_stockfish: chessrl.Player = Stockfish(
        chess.WHITE,
        stockfish_binary,
        elo=white_elo,
    )
    black_stockfish = Stockfish(
        chess.BLACK,
        stockfish_binary,
        elo=black_elo,
    )
    # black_stockfish: chessrl.Player = Agent(
    #     chess.BLACK,
    #     stockfish_binary=stockfish_binary,
    # ) # for checking agent structure is working properly

    board = game.get_new_board()

    # play out game
    while game.get_result(board) is None:
        game.next_move(
            board,
            white_stockfish,
            black_stockfish,
        )

    dataset.append(board)
    if tqbar is not None:
        tqbar.update(1)

    # kill stockfish processes
    white_stockfish.close()
    black_stockfish.close()


def gen_data(
    stockfish_binary: str,
    save_path: str,
    num_games: int = 100,
    workers: int = 2,
):
    """Generate a dataset of games by playing Stockfish against itself using\
    distributed workers."""
    logger = Logger.get_instance()
    pbar = tqdm(total=num_games)

    d = GameDataset()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(num_games):
            executor.submit(
                play_game,
                stockfish_binary=stockfish_binary,
                dataset=d,
                tqbar=pbar,
            )
    # play_game(stockfish_binary=stockfish_binary, dataset=d, tqbar=pbar)
    pbar.close()

    logger.info("Saving dataset...")
    d.save(save_path)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Plays some chess games using Stockfish instances and stores the results as a dataset."
    )
    parser.add_argument(
        "stockfish_binary",
        metavar="stockfish binary",
        help="Stockfish binary path",
    )
    parser.add_argument(
        "data_path",
        metavar="datadir",
        help="Path of JSON dataset.",
    )
    parser.add_argument(
        "--games",
        metavar="games",
        type=int,
        default=10,
        help="Number of games to play. Default 10.",
    )
    parser.add_argument(
        "--workers",
        metavar="workers",
        type=int,
        default=2,
        help="Number of workers for games. Default 2.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Log debug messages on screen. Default false.",
    )

    args = parser.parse_args()

    logger = Logger.get_instance()
    logger.set_level(1)

    if args.debug:
        logger.set_level(0)

    gen_data(args.stockfish_binary, args.data_path, args.games, args.workers)


if __name__ == "__main__":
    main()
