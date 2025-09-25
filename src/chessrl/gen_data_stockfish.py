from gamestockfish import GameStockfish
from stockfish import Stockfish
from dataset import DatasetGame
from game import Game
from lib.logger import Logger
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import argparse
import numpy as np


def play_game(stockfish_bin, dataset, depth=10, tqbar=None):

    is_white = Game.WHITE if np.random.random() <= .5 else Game.BLACK

    g = GameStockfish(stockfish=stockfish_bin,
                      player_color=is_white,
                      stockfish_depth=depth)
    stockf = Stockfish(is_white, stockfish_bin, depth)

    # first move
    bm = stockf.get_move(g, first_move=True)
    g.move(bm)

    # play out game
    while g.get_result() is None:
        bm = stockf.get_move(g)
        g.move(bm)

    dataset.append(g)
    if tqbar is not None:
        tqbar.update(1)

    # Kill stockfish processes
    g.tearup()
    stockf.kill()


def gen_data(stockfish_bin, save_path, num_games=100, workers=2):
    logger = Logger.get_instance()
    d = DatasetGame()
    pbar = tqdm(total=num_games)

    # TODO: add sampling of ELOs to make games different

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(num_games):
            executor.submit(play_game,
                            stockfish_bin=stockfish_bin,
                            dataset=d,
                            tqbar=pbar)
    pbar.close()
    logger.info("Saving dataset...")
    d.save(save_path)


def main():
    parser = argparse.ArgumentParser(description="Plays some chess games,"
                                     "stores the result and trains a model.")
    parser.add_argument('stockfish_bin', metavar='stockbin',
                        help="Stockfish binary path")
    parser.add_argument('data_path', metavar='datadir',
                        help="Path of .JSON dataset.")
    parser.add_argument('--games', metavar='games', type=int,
                        default=10)
    parser.add_argument('--workers', metavar='workers', type=int,
                        default=2, help="Number of workers for games.")
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help="Log debug messages on screen. Default false.")

    args = parser.parse_args()

    logger = Logger.get_instance()
    logger.set_level(1)

    if args.debug:
        logger.set_level(0)

    gen_data(args.stockfish_bin, args.data_path, args.games, args.workers)


if __name__ == "__main__":
    main()
