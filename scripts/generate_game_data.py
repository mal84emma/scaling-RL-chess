"""Generate training games by playing two Stockfish models against each other."""

import argparse
from concurrent.futures import ThreadPoolExecutor

import chess
from tqdm import tqdm

import chessrl
import chessrl.game as game
from chessrl import Agent, GameDataset, Stockfish
from chessrl.utils import Logger


def play_game(stockfish_binary, dataset, tqbar=None):
    """ToDo."""
    # TODO: add sampling of ELOs to make games different
    white_stockfish: chessrl.Player = Stockfish(
        chess.WHITE,
        stockfish_binary,
        elo=3000,
    )
    # black_stockfish = Stockfish(chess.BLACK, stockfish_binary)
    black_stockfish: chessrl.Player = Agent(
        chess.BLACK,
        stockfish_binary=stockfish_binary,
    )

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

    # Kill stockfish processes
    white_stockfish.close()
    black_stockfish.close()


def gen_data(stockfish_binary, save_path, num_games=100, workers=2):
    """ToDo."""
    logger = Logger.get_instance()
    d = GameDataset()
    pbar = tqdm(total=num_games)

    # TODO: add sampling of ELOs to make games different

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
        description="Plays some chess games,stores the result and trains a model."
    )
    parser.add_argument(
        "stockfish_binary", metavar="stockbin", help="Stockfish binary path"
    )
    parser.add_argument("data_path", metavar="datadir", help="Path of .JSON dataset.")
    parser.add_argument("--games", metavar="games", type=int, default=10)
    parser.add_argument(
        "--workers",
        metavar="workers",
        type=int,
        default=2,
        help="Number of workers for games.",
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
