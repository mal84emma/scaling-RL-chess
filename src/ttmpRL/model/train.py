"""ToDo."""

from ttmpRL.dataset import GameDataset

from .dataloader import GameDataSequence
from .model import ChessModel


def train_model(
    model: ChessModel,
    dataset: GameDataset,
    epochs=1,
    logdir=None,
    batch_size=1,
    validation_split=0,
):
    """Trains the model using previous recorded games."""
    if len(dataset) <= 0:
        return

    if validation_split > 0:
        split_point = len(dataset) - int(validation_split * len(dataset))

        games_train = GameDataset(dataset[:split_point])
        games_val = GameDataset(dataset[split_point:])
        val_gen = GameDataSequence(games_val, batch_size=batch_size)
    else:
        games_train = dataset
        val_gen = None

    train_gen = GameDataSequence(games_train, batch_size=batch_size, random_flips=0.1)

    model.train_generator(train_gen, epochs=epochs, logdir=logdir, val_gen=val_gen)
