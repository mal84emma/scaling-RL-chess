"""ToDo."""

from chessrl.dataset import GameDataset

from .dataloader import GameDataSequence
from .model import ChessModel


# useful to have as a function as I'll use it at test time
def train_model(
    model: ChessModel,
    dataset: GameDataset,  # this won't be a GameDataset any more, likely a list of position-score pairs
    epochs=1,
    logdir=None,
    batch_size=1,
    validation_split=0,
):
    """Trains the value model using (board position, cp_score) examples."""
    assert len(dataset) > 0, "You must provide some training examples. None provided."

    if validation_split > 0:
        split_point = len(dataset) - int(validation_split * len(dataset))

        games_train = dataset[:split_point]
        games_val = dataset[split_point:]
        val_gen = GameDataSequence(games_val, batch_size=batch_size)  # adjust this
    else:
        games_train = dataset
        val_gen = None

    train_gen = GameDataSequence(
        games_train, batch_size=batch_size, random_flips=0.1
    )  # adjust this

    model.train_generator(train_gen, epochs=epochs, logdir=logdir, val_gen=val_gen)
