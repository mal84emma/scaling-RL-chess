__all__ = ("PositionDataSequence",)

import numpy as np
from keras.utils import Sequence

from chessrl.dataset import PositionDataset

from .encoder import get_game_state


class PositionDataSequence(Sequence):
    """Transforms a dataset to a data generator to be fed to the training
    loop of the neural network.

    Attributes:
        dataset: PositionDataset. Dataset of game_state-score pairs.
        batch_size: int. # of board representations of each batch.
    """

    def __init__(
        self,
        dataset: PositionDataset,
        batch_size: int = 10,
    ):
        self.dataset = dataset
        self.batch_size = min(batch_size, len(dataset))

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx):
        game_states, scores = self.dataset[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_x = [get_game_state(state) for state in game_states]  # board reprs
        batch_y = scores  # target scores

        return np.asarray(batch_x), np.asarray(batch_y)
