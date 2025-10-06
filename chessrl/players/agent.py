import numpy as np

import chessrl.players.netencoder as netencoder

from game import Game
from chessrl.agents.player import Player
from chessrl.agents.model import ChessModel
from chessrl.utils.dataset import DatasetGame


class Agent(Player):
    """ AI agent which will play thess.

    Parameters:
        model: Model. Model encapsulating the neural network operations.
        move_encodings: list. List of all possible uci movements.
        uci_dict: dict. Dictionary with mappings 'uci'-> int. It's used
        to predict the policy only over the legal movements.
    """
    def __init__(self, color, weights_path=None):
        super().__init__(color)

        self.model = ChessModel(compile_model=True, weights=weights_path)
        self.move_encodings = netencoder.get_uci_labels()
        self.uci_dict = {u: i for i, u in enumerate(self.move_encodings)}

    def get_move(self, game:Game) -> str:
        """ Finds and returns the best possible move (UCI encoded)

        Parameters:
            game: Game. Current game before the move of this agent is made.
            real_game: Whether to use MCTS or only the neural network (self
            play vs playing in a real environment).
            max_iters: if not playing a real game, the max number of iterations
            of the MCTS algorithm.

        Returns:
            str. UCI encoded movement.
        """
        move = Game.NULL_MOVE
        policy = self.predict_policy(game)
        move = game.get_legal_moves()[np.argmax(policy)]
        return move

    def predict_outcome(self, game:Game) -> float:
        """ Predicts the outcome of a game from the current position """
        game_matr = netencoder.get_game_state(game)
        return self.model.predict(np.expand_dims(game_matr, axis=0))[1][0][0]

    def predict_policy(self, game:Game, mask_legal_moves=True) -> float:
        """ Predict the policy distribution over all possible moves. """
        game_matr = netencoder.get_game_state(game)
        policy = self.model.predict(np.expand_dims(game_matr, axis=0))[0][0]
        if mask_legal_moves:
            legal_moves = game.get_legal_moves()
            policy = [policy[self.uci_dict[x]] for x in legal_moves]
        return policy

    # TODO: why is this a method of Agent? This should be a script.
    # Separate the NN model from the Agent class. Rewrite it so that
    # the Agent can also validly take position score predictions from   
    # Stockfish or other engines.
    def train(self, dataset: DatasetGame,
              epochs=1, logdir=None, batch_size=1,
              validation_split=0):
        """ Trains the model using previous recorded games."""
        if len(dataset) <= 0:
            return

        if validation_split > 0:
            split_point = len(dataset) - int(validation_split * len(dataset))

            games_train = DatasetGame(dataset[:split_point])
            games_val = DatasetGame(dataset[split_point:])
            val_gen = netencoder.DataGameSequence(games_val,
                                                  batch_size=batch_size)
        else:
            games_train = dataset
            val_gen = None

        train_gen = netencoder.DataGameSequence(games_train,
                                                batch_size=batch_size,
                                                random_flips=.1)

        self.model.train_generator(train_gen,
                                   epochs=epochs,
                                   logdir=logdir,
                                   val_gen=val_gen)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

    def clone(self, ):
        return Agent(self.color, self.model.weights_path)
