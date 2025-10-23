"""Neural network model used by the agent to estimate the position score."""

__all__ = ("ChessScoreModel",)

import chess
import numpy as np
from keras import Model, callbacks, layers, ops
from keras import backend as K
from keras.losses import mean_absolute_error, mean_squared_error
from keras.optimizers import Adam, Nadam
from keras.utils import Sequence

from .encoder import get_game_state


def _lr_scheduler(epoch, lr):
    """Learning rate scheduler."""
    if epoch > 0:
        return lr * 0.95
    return lr


class ChessScoreModel:
    """Neural Network model for scoring encoded chess positions."""

    def __init__(
        self,
        model_architecture: str = "cnn",
        compile_model: bool = True,
        weights: str = None,
    ):
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

        inp = layers.Input((8, 8, 18))

        if model_architecture == "cnn":
            op = self._build_cnn(inp)
        elif model_architecture == "transformer":
            op = self._build_transformer(inp)
        else:
            raise ValueError(f"Unknown model architecture: {model_architecture}")

        self.model: Model = Model(
            inputs=inp,
            outputs=op,
        )

        self.weights_path = weights
        if weights:
            self.model.load_weights(weights)

        if compile_model:
            self.model.compile(
                Nadam(
                    learning_rate=1e-3,  # Standard is 1e-3 - previously 1e-5 has been good
                    # weight_decay=1e-4,  # Add weight decay for generalization
                ),
                loss=["mean_absolute_error"],  # "mean_squared_error"
                metrics=["accuracy", "mse", "mae"],  # Added MAE for better monitoring
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
        cbacks = []
        if logdir is not None:
            tensorboard_callback = callbacks.TensorBoard(
                log_dir=logdir, histogram_freq=0, write_graph=False, update_freq=1
            )
            cbacks.append(tensorboard_callback)

        cbacks.append(
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                min_delta=1,  # Only continue if improvement is meaningful
                mode="min",
            )
        )
        # cbacks.append(
        #     ReduceLROnPlateau(
        #         monitor="val_loss",
        #         factor=0.5,  # Reduce LR by half when plateau
        #         patience=5,  # Wait 5 epochs before reducing
        #         min_lr=1e-6,  # Don't go below this learning rate
        #         verbose=1,
        #     )
        # )
        cbacks.append(
            callbacks.LearningRateScheduler(_lr_scheduler, verbose=1)
        )  # gradually reducing lr seems to improve training
        cbacks.append(callbacks.BackupAndRestore(backup_dir="data/models/train_backup"))

        # train model
        self.model.fit(
            generator,
            epochs=epochs,
            validation_data=val_gen,
            verbose=1,
            callbacks=cbacks,
        )

    def __del__(self):
        K.clear_session()

    def __loss(self, y_true, y_pred):
        # return mean_squared_error(y_true, y_pred)
        return mean_absolute_error(y_true, y_pred)

    def _build_cnn(self, inp):
        """Builds CNN model architecture.

        Following model architecture from
        https://github.com/Zeta36/chess-alpha-zero/blob/master/model.png
        Actaully another source has a slightly different architecture
        https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/
        """

        x = layers.Conv2D(
            data_format="channels_last",
            filters=256,
            kernel_size=3,
            strides=1,
            padding="same",  # 'same' padding is odd, but apprently what people use
            kernel_regularizer="l2",
        )(inp)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation("relu")(x)

        for _ in range(7):  # originally 10 res blocks
            x = self.__res_block(x)

        # Value Head
        # ==========
        val_head = layers.Conv2D(
            data_format="channels_last",
            filters=1,
            strides=1,
            kernel_size=1,
            padding="same",  # "valid" used in original implementation
            kernel_regularizer="l2",
        )(x)
        val_head = layers.BatchNormalization(axis=-1)(val_head)
        val_head = layers.Activation("relu")(val_head)
        val_head = layers.Flatten()(val_head)
        val_head = layers.Dense(
            256,
            kernel_regularizer="l2",
            activation="relu",
        )(val_head)
        # val_head = Dropout(0.1)(val_head)  # Add dropout for regularization
        val_head = layers.Dense(
            1,
            kernel_regularizer="l2",
            activation="tanh",
            name="value_out",
        )(val_head)
        val_head = layers.Rescaling(1500)(val_head)  # scale tanh output to centipawns

        return val_head

    def __res_block(self, block_input):
        """Builds a residual block"""
        x = layers.Conv2D(
            data_format="channels_last",
            filters=256,
            kernel_size=3,
            padding="same",
            strides=1,
            kernel_regularizer="l2",
        )(block_input)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(
            data_format="channels_last",
            filters=256,
            kernel_size=3,
            padding="same",
            strides=1,
            kernel_regularizer="l2",
        )(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Add()([block_input, x])
        x = layers.Activation("relu")(x)
        return x

    def _build_transformer(self, inp):
        """Builds Transformer model architecture.

        Following implementation example from
        https://keras.io/examples/timeseries/timeseries_classification_transformer/
        and taking inspiration for the model architecture from
        https://arxiv.org/pdf/2409.12272 (see Appendix A)
        """
        x = layers.Reshape((64, 18))(inp)  # input vector of features for each sqaure
        x = BoardEmbedding()(x)  # feature & positional embedding

        for _ in range(2):
            x = self.__transformer_encoder(x)

        # x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        # OR
        x = layers.Conv1D(filters=1, kernel_size=1, kernel_regularizer="l2")(x)
        x = layers.Activation("mish")(x)
        x = layers.Flatten()(x)
        # OR
        # x = LinearProjection()(x)
        # x = layers.Flatten()(x)

        x = layers.Dense(256, kernel_regularizer="l2", activation="mish")(x)
        x = layers.Dense(1, "tanh", kernel_regularizer="l2", name="value_out")(x)
        x = layers.Rescaling(1500)(x)  # scale tanh output to centipawns

        return x

    def __transformer_encoder(
        self,
        inputs,
        head_size=64,
        num_heads=4,
        ff_dim=16,
        dropout=0,
    ):
        # Attention and Normalization
        x = layers.MultiHeadAttention(
            key_dim=head_size,
            num_heads=num_heads,
            dropout=dropout,
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = layers.Add()([x, inputs])

        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="mish")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Add()([x, res])
        return x


class LinearProjection(layers.Layer):
    """Apply learnable projection matrix element-wise to each token."""

    def __init__(self, output_size=32):
        super().__init__()
        self.output_size = output_size

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.output_size),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        return ops.matmul(inputs, self.w)


class BoardEmbedding(layers.Layer):
    def __init__(self, output_size: int = 10):  # prev 8
        super().__init__()
        self.output_size = output_size

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.output_size),
            initializer="random_normal",
            trainable=True,
        )
        self.a = self.add_weight(
            shape=(
                input_shape[-2],
                self.output_size,
            ),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(
                input_shape[-2],
                self.output_size,
            ),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        # not sure if this is right
        return ops.multiply(self.a, ops.matmul(inputs, self.w)) + self.b
