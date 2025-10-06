import argparse
import os

from agent import Agent
from dataset import DatasetGame
from game import Game
from lib.logger import Logger
from lib.model import get_model_path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def train(model_dir, dataset_path, epochs=1, batch_size=8):
    """Loads (or creates, if not found) a model from model_dir, trains it
    and saves the results.

    Parameters:
        model_dir: str. Directory which contains the model
        dataset_string: str. DatasetGame serialized as a string.
    """
    logger = Logger.get_instance()

    logger.info("Loading dataset")

    data_train = DatasetGame()
    data_train.load(dataset_path)

    if not os.path.exists(model_dir):
        logger.info(f"Model directory {model_dir} does not exist yet. Creating it...")
        os.mkdir(model_dir)
    model_path = get_model_path(model_dir)

    logger.info("Loading the agent...")
    chess_agent = Agent(color=Game.WHITE)
    try:
        chess_agent.load(model_path)
    except OSError:
        logger.warning("Model not found, training a fresh one.")
    chess_agent.train(
        data_train,
        logdir=model_dir,
        epochs=epochs,
        validation_split=0.25,
        batch_size=batch_size,
    )
    logger.info("Saving the agent...")
    new_model_path = get_model_path(model_dir, increment=True)
    chess_agent.save(new_model_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plays some chess games,stores the result and trains a model."
    )
    parser.add_argument(
        "model_dir",
        metavar="modeldir",
        help="where to store (and load from)the trained model and the logs",
    )
    parser.add_argument("data_path", metavar="datadir", help="Path of .JSON dataset.")
    parser.add_argument("--epochs", metavar="epochs", type=int, default=1)
    parser.add_argument(
        "--bs", metavar="bs", help="Batch size. Default 8", type=int, default=8
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

    train(args.model_dir, args.data_path, args.epochs, args.bs)


if __name__ == "__main__":
    main()
