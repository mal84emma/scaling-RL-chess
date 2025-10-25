import argparse
import os

from chessrl import PositionDataset
from chessrl.model import ChessScoreModel, get_model_path, train_model
from chessrl.utils import Logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


def train(
    model_dir: str,
    dataset_path: str,
    epochs: int = 1,
    batch_size: int = 8,
    val_split: float = 0.2,
) -> None:
    """Loads (or creates, if not found) a model from model_dir, trains it
    and saves the results.
    """
    logger = Logger.get_instance()

    logger.info("Loading dataset")
    data_train = PositionDataset()
    data_train.load(dataset_path)

    if not os.path.exists(model_dir):
        logger.info(f"Model directory {model_dir} does not exist yet. Creating it...")
        os.mkdir(model_dir)
    model_path = get_model_path(model_dir)

    logger.info("Loading the model...")
    model = ChessScoreModel(training_mode=True)

    try:
        model.load_weights(model_path)
    except OSError:
        logger.warning("Model not found, training a fresh one.")

    train_model(
        model,
        data_train,
        epochs=epochs,
        batch_size=batch_size,
        logdir=model_dir,
        validation_split=val_split,
    )

    logger.info("Saving the model...")
    new_model_path = get_model_path(model_dir, increment=True)
    model.save_weights(new_model_path)

    return


def main():
    parser = argparse.ArgumentParser(
        description="Plays some chess games,stores the result and trains a model."
    )
    parser.add_argument(
        "model_dir",
        metavar="modeldir",
        help="where to store (and load from)the trained model and the logs",
    )
    parser.add_argument(
        "data_path",
        metavar="datadir",
        help="Path of .JSON dataset.",
    )
    parser.add_argument(
        "--epochs",
        metavar="epochs",
        type=int,
        default=1,
        help="Number of training epochs. Default 1",
    )
    parser.add_argument(
        "--bs",
        metavar="batch size",
        type=int,
        default=10,
        help="Batch size. Default 10",
    )
    parser.add_argument(
        "--vs",
        metavar="batch size",
        type=float,
        default=0.2,
        help="Validation split. Default 0.2",
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

    train(args.model_dir, args.data_path, args.epochs, args.bs, args.vs)


if __name__ == "__main__":
    main()
