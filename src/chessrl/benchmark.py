""" This script serves as a way to get tangible metrics about how well a
trained agent behaves agaisnt Stockfish. For that, several games are played
and a summary of them is returned.
"""

from agent import Agent
from ttmpt import TTAgent
from gamestockfish import GameStockfish
from timeit import default_timer as timer
import argparse
from lib.logger import Logger
from lib.model import get_model_path
from concurrent.futures import ProcessPoolExecutor

import random
import os
import traceback
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def process_initializer():
    """ Initializer of the training threads in in order to detect if there
    is a GPU available and use it. This is needed to initialize TF inside the
    child process memory space."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.keras.backend.clear_session()


def play_game_job(id: int, model_path, stockfish_depth, stockfish_quality=1.0,
                  log=False, use_ttmp=False, plot=False, delay=0.5):
    """ Plays a game and returns the result..

    Parameters:
        id: Play ID (i.e. worker ID).
        model_path: path to the .weights.h5 model. If it not exists, it will play with
        a fresh one.
        stockfish_depth: int. Difficulty of stockfish.
    """
    logger = Logger.get_instance()

    agent_is_white = True if random.random() <= .5 else False

    game_env = GameStockfish(player_color=agent_is_white,
                             stockfish='../../res/stockfish-17-macos-m1-apple-silicon',
                             stockfish_depth=stockfish_depth,
                             stockfish_quality=stockfish_quality)

    try:
        agent_type = Agent if not use_ttmp else TTAgent
        chess_agent = agent_type(color=agent_is_white, weights_path=model_path)
    except OSError:
        logger.error("Model not found. Exiting.")
        return None

    if log:
        logger.info(f"Starting game {id}: Agent is {'white' if agent_is_white else 'black'}")

    if plot:
        fig,ax = plt.subplots(1,1)
        img = game_env.plot_board(return_img=True, show_moves=False)
        im = ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        plt.pause(delay)

    try:
        timer_start = timer()

        if not agent_is_white:
            game_env.move('00000') # make stockfish take first move

        while game_env.get_result() is None:

            agent_move = chess_agent.best_move(game_env, real_game=True)
            game_env.move(agent_move)

            if plot:
                im.set_data(game_env.plot_board(return_img=True, show_moves=True))
                fig.canvas.draw_idle()
                plt.pause(delay)
        timer_end = timer()

        if log:
            logger.info(f"Game {id} done. Result: {game_env.get_result()}. "
                        f"took {round(timer_end-timer_start, 2)} secs")
    except Exception:
        logger.error(traceback.format_exc())

    res = {'color': agent_is_white, 'result': game_env.get_result()}
    game_env.tearup()
    if plot: plt.close(fig)
    return res


def benchmark(
        model_dir,
        workers=1,
        games=10,
        stockfish_depth=5,
        stockfish_quality=1.0,
        log=False,
        plot=True,
        delay=0.5,
        distributed=False,
        use_ttmp=False
    ):
    """ Plays N games and gets stats about the results.

    Parameters:
        model_dir: str. Directory where the neural net weights and training
            logs will be saved.
        workers: number of concurrent games (workers which will play the games)
    """
    multiprocessing.set_start_method('spawn', force=True)

    if log:
        logger = Logger.get_instance()
        if distributed:
            logger.info(f"Setting up {workers} concurrent games.")

    model_path = get_model_path(model_dir)

    if distributed:
        with ProcessPoolExecutor(workers, initializer=process_initializer)\
                as executor:

            results = []
            for i in range(games):
                results.append(executor.submit(play_game_job, *[i,
                                                                model_path,
                                                                stockfish_depth]))
        results = [r.result() for r in results]

    else:
        results = [play_game_job(i, model_path, 
                                 stockfish_depth, stockfish_quality,
                                 log, use_ttmp, plot=plot, delay=delay)\
                                    for i in tqdm(range(games), desc="Games played")]

    if log:
        logger.debug("Calculating stats.")
    won = [1
           if x['color'] is True and x['result'] == 1
           or x['color'] is False and x['result'] == -1 else 0  # noqa:W503
           for x in results]

    if log:
        print("##################### SUMMARY ###################")
        print(f"Games played: {games}")
        print(f"Games won: {len([x for x in won if x == 1])}")
        print(f"Games drawn: {len([x for x in results if x['result'] == 0])}")
        print("#################################################")

    return dict(played=games, won=len([x for x in won if x == 1]),
                drawn=len([x for x in results if x['result'] == 0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plays some chess games against"
                                     "Stockfish and reports the results.")
    parser.add_argument('model_dir', metavar='modeldir',
                        help="where to store (and load from)"
                        "the trained model and the logs")
    parser.add_argument('--plot', metavar='plot', type=bool,
                        default=True)
    parser.add_argument('--delay', metavar='delay', type=float,
                        default=0.5)

    args = parser.parse_args()

    benchmark(args.model_dir, workers=2, log=True,
              plot=args.plot, delay=args.delay,
              distributed=False, use_ttmp=True,
              stockfish_depth=5, games=10,
              stockfish_quality=0.75)
    # see page 77 of https://web.ist.utl.pt/diogo.ferreira/papers/ferreira13impact.pdf
    # for comparison between stockfish depth and Elo rating
    # note: I've put in random stockfish depth just to make the games different
