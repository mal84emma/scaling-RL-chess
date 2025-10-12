__all__ = ("GameDataSequence",)

import numpy as np
from keras.utils import Sequence, to_categorical

import chessrl
from chessrl.dataset import GameDataset

from .encoder import get_game_state

# need to figure out how to handle both training data generated from simulated
# stockfish games, and training data that is generated at test-time which only
# has position-score information.
# this probably means I want to do a pre-processing of the training games,
# maybe in the training script that preps them into a data loader, playing out
# the games, scoring positions, and loading the position-score pairs into the
# data loader.


class GameDataSequence(Sequence):
    """Transforms a Dataset to a Data generator to be fed to the training
    loop of the neural network.

    Attributes:
        dataset: GameDataset. Dataset of fames
        batch_size: int. Nb of board representations of each batch
        uci_ids: dict. Encoding the move UCI labels to one-hot.
        random_flips: float. Proportion of board representation which will
                        be flipped 180 degrees.
    """

    def __init__(self, dataset: GameDataset, batch_size: int = 8, random_flips=0):
        self.dataset = dataset
        self.batch_size = min(batch_size, len(dataset))
        self.uci_ids = {u: i for i, u in enumerate(chessrl.UCI_MOVES)}
        self.random_flips = random_flips

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.dataset[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = []  # Board reprs
        batch_y_policies = []
        batch_y_values = []

        for i in batch:
            i_augmented = self.dataset.augment_game(i)

            flip = np.random.rand() < self.random_flips
            batch_x.extend(
                [get_game_state(i_g["game"], flipped=flip) for i_g in i_augmented]
            )
            batch_y_policies.extend(
                [
                    to_categorical(self.uci_ids[targets["next_move"]], num_classes=1968)
                    for targets in i_augmented
                ]
            )
            batch_y_values.extend([targets["result"] for targets in i_augmented])

        return np.asarray(batch_x), (
            np.asarray(batch_y_policies),
            np.asarray(batch_y_values),
        )
