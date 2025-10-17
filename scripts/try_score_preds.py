"""Visually check score predictions from trained model."""

import chess
import numpy as np

from chessrl import PositionDataset
from chessrl.model import ChessScoreModel, get_game_state, get_model_path


def main(n_positions: int = 10):
    data_train = PositionDataset()
    data_train.load("data/positions/positions.json")

    model_path = get_model_path("data/models/modela")
    model = ChessScoreModel()
    model.load_weights(model_path)

    for i in range():
        state, score = data_train[i]
        x = get_game_state(state)
        pred = model.predict(np.expand_dims(x, axis=0))
        print(f"\nBoard {i}:")
        print(state)
        print(chess.COLOR_NAMES[state.turn] + " to move")
        print(f"Predicted score: {pred[0][0]}")
        print(f"True score: {score}")


if __name__ == "__main__":
    main(n_positions=10)
