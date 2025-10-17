"""Neural network model used by the agent to estimate the position score."""

__all__ = ("ChessScoreModel",)

import chess
import numpy as np
from keras import Model
from keras import backend as K
from keras.callbacks import (
    BackupAndRestore,
    EarlyStopping,
    # LearningRateScheduler,
    TensorBoard,
)
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Rescaling,
)
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.utils import Sequence

from .encoder import get_game_state


def _lr_scheduler(epoch, lr):
    """Learning rate scheduler."""
    if epoch > 0:
        return lr * 0.95
    return lr


class ChessScoreModel:
    """Neural Network model for scoring encoded chess positions."""

    def __init__(self, compile_model: bool = True, weights: str = None):
        """Creates the model. This code builds a ResNet that will act as both
        the policy and value network (see AlphaZero paper for more info).

        Parameters:
            compile_model: bool. Whether the model will be compiled on creation
            weights: str. Path to the neural network weights. After the
                    creation, the NN will load the weights under that path.

        Attributes:
            model: Neural net model.
            __gra = TF Graph. You should not use this externally.
        """
        # following model architecture from
        # https://github.com/Zeta36/chess-alpha-zero/blob/master/model.png
        # actaully another source has a slightly different architecture
        # https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/

        inp = Input((8, 8, 18))

        x = Conv2D(
            data_format="channels_last",
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",  # 'same' padding is odd, but apprently what people use
            kernel_regularizer="l2",
        )(inp)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        for _ in range(7):  # originally 10 res blocks
            x = self.__res_block(x)

        # Value Head
        # ==========
        # this can be trained on Stockfish evaluation of position
        # best to do for 'typical' games, either from Lichess or from
        # simulated games with easier engines
        # you don't need to learn all of chess, you just need to learn
        # the positions that you are likely to encounter (Lichess has these
        # with their stockfish scores, or we can generate them ourselves)

        val_head = Conv2D(
            data_format="channels_last",
            filters=1,
            strides=1,
            kernel_size=1,
            padding="same",  # "valid" used in original implementation
            kernel_regularizer="l2",
        )(x)
        val_head = BatchNormalization(axis=-1)(val_head)
        val_head = Activation("relu")(val_head)
        val_head = Flatten()(val_head)
        val_head = Dense(256, kernel_regularizer="l2", activation="relu")(val_head)
        val_head = Dense(
            1, kernel_regularizer="l2", activation="tanh", name="value_out"
        )(val_head)
        val_head = Rescaling(1500)(val_head)  # scale tanh output to centipawns

        self.model: Model = Model(
            inputs=inp,
            outputs=val_head,
        )

        self.weights_path = weights
        if weights:
            self.model.load_weights(weights)

        if compile_model:
            self.model.compile(
                Adam(
                    learning_rate=0.0001
                ),  # may need tuning - actually very hard to tune and important to get right, goodness also seems affected by batch size
                loss=["mean_squared_error"],
                metrics=["accuracy", "mse"],
            )

    def predict(self, inp):
        return self.model.predict(inp, verbose=None)

    def score_position(self, board: chess.Board):
        encoded_board = get_game_state(board)
        encoded_board = np.expand_dims(encoded_board, axis=0)
        return self.model.predict(encoded_board, verbose=None)[0][0]

    def load_weights(self, weights_path: str):
        self.model.load_weights(weights_path)
        self.weights_path = weights_path

    def save_weights(self, weights_path: str):
        self.model.save_weights(weights_path)
        self.weights_path = weights_path

    def train(self, game_state, game_outcome, next_action):
        # pass
        raise NotImplementedError("Use train_generator instead.")

    def train_generator(
        self,
        generator: Sequence,
        val_gen: Sequence = None,
        epochs: int = 1,
        logdir: str = None,
    ):
        """Train model using generator(s) of position-score data."""

        # set up callbacks
        callbacks = []
        if logdir is not None:
            tensorboard_callback = TensorBoard(
                log_dir=logdir, histogram_freq=0, write_graph=False, update_freq=1
            )
            callbacks.append(tensorboard_callback)

        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        )
        # callbacks.append(LearningRateScheduler(_lr_scheduler, verbose=1))
        callbacks.append(BackupAndRestore(backup_dir="data/models/train_backup"))

        # train model
        self.model.fit(
            generator,
            epochs=epochs,
            validation_data=val_gen,
            verbose=1,
            callbacks=callbacks,
        )

    def __del__(self):
        K.clear_session()

    def __loss(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def __res_block(self, block_input):
        """Builds a residual block"""
        x = Conv2D(
            data_format="channels_last",
            filters=256,
            kernel_size=3,
            padding="same",
            strides=1,
            kernel_regularizer="l2",
        )(block_input)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
        x = Conv2D(
            data_format="channels_last",
            filters=256,
            kernel_size=3,
            padding="same",
            kernel_regularizer="l2",
        )(x)
        x = BatchNormalization(axis=-1)(x)
        x = Add()([block_input, x])
        x = Activation("relu")(x)
        return x
