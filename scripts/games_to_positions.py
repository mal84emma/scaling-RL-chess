"""Convert dataset of games to dataset of positions."""

import argparse
import random

import chess
import numpy as np
from tqdm import tqdm

import chessrl.game as game
from chessrl.dataset import GameDataset, PositionDataset
from chessrl.scorer import StockfishScorer


def convert_games_to_positions(
    input_path: str,
    output_path: str,
    n_positions: int,
    stockfish_binary: str,
):
    """Extract `n_positions` positions from dataset of games, score them
    with Stockfish, and save as a dataset of positions."""
    games = GameDataset()
    games.load(input_path)

    positions = PositionDataset()

    scorer = StockfishScorer(stockfish_binary)

    ngames = len(games)
    div = n_positions // ngames
    rem = n_positions - (div * ngames)
    n_positions_per_game = np.ones(ngames, dtype=int) * div
    n_positions_per_game[:rem] += 1

    for j, g in tqdm(
        enumerate(games.games),
        total=ngames,
        desc="Getting positions from games...",
    ):
        tmp_board = chess.Board()

        posn_indices = random.sample(
            range(len(g.move_stack)),
            k=n_positions_per_game[j],
        )

        for i, move in enumerate(game.get_history(g)["moves"]):
            game.move(tmp_board, move)

            if i in posn_indices:
                score = scorer.score_position(tmp_board)
                positions.add_position(tmp_board, score)

    scorer.close()

    print(f"Extracted {len(positions)} positions.")
    print("Saving dataset...")
    positions.shuffle()
    positions.save(output_path)

    return


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Plays some chess games, stores the result and trains a model."
    )
    parser.add_argument(
        "input_path", metavar="datadir", help="Path of .JSON dataset to convert."
    )
    parser.add_argument(
        "output_path", metavar="outdir", help="Path of .JSON for saving."
    )
    parser.add_argument(
        "npos",
        metavar="# positions",
        type=int,
    )
    parser.add_argument(
        "stockfish_binary", metavar="stockbin", help="Stockfish binary path"
    )

    args = parser.parse_args()

    convert_games_to_positions(
        args.input_path,
        args.output_path,
        args.npos,
        args.stockfish_binary,
    )


if __name__ == "__main__":
    main()
    # e.g. python scripts/games_to_positions.py data/games.json data/positions.json 1000 resources/stockfish-17-macos-m1-apple-silicon
