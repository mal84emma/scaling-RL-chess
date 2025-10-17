"""Helper functions for training score prediction model using position-score dataset."""

from chessrl.dataset import PositionDataset

from .dataloader import PositionDataSequence
from .model import ChessScoreModel


def train_model(
    model: ChessScoreModel,
    dataset: PositionDataset,
    epochs: int = 1,
    batch_size: int = 10,
    validation_split: float = 0,
    logdir: str = None,
):
    """Trains the value model using (board position, centipawn score) examples."""
    assert len(dataset) > 0, "You must provide some training examples. None provided."

    if validation_split > 0:
        split_point = len(dataset) - int(validation_split * len(dataset))

        positions_train = PositionDataset(*dataset[:split_point])
        positions_val = PositionDataset(*dataset[split_point:])
        val_gen = PositionDataSequence(positions_val, batch_size=batch_size)
    else:
        positions_train = dataset
        val_gen = None

    train_gen = PositionDataSequence(positions_train, batch_size=batch_size)

    model.train_generator(train_gen, epochs=epochs, logdir=logdir, val_gen=val_gen)

    return
