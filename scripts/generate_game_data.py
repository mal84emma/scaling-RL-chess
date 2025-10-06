import chess
from ttmpRL import Game, Stockfish
from ttmpRL.dataset import GameDataset
from ttmpRL.utils import Logger

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import numpy as np


def play_game(stockfish_bin, dataset, tqbar=None):

    # TODO: add sampling of ELOs to make games different
    white_stockfish = Stockfish(chess.WHITE, stockfish_bin)
    black_stockfish = Stockfish(chess.BLACK, stockfish_bin)

    game = Game(white_player=white_stockfish,
             black_player=black_stockfish)

    # play out game
    while game.get_result() is None:
        game.move()

    dataset.append(game)
    if tqbar is not None:
        tqbar.update(1)

    # Kill stockfish processes
    white_stockfish.close()
    black_stockfish.close()


def gen_data(stockfish_bin, save_path, num_games=100, workers=2):
    logger = Logger.get_instance()
    d = GameDataset()
    pbar = tqdm(total=num_games)

    # TODO: add sampling of ELOs to make games different

    #with ThreadPoolExecutor(max_workers=workers) as executor:
    # for _ in range(num_games):
    #     executor.submit(play_game,
    #                     stockfish_bin=stockfish_bin,
    #                     dataset=d,
    #                     tqbar=pbar)
    # pbar.close()

    for _ in tqdm(range(num_games)):
        play_game(stockfish_bin=stockfish_bin,
                  dataset=d,
                  tqbar=pbar)

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
