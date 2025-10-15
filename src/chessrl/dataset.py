__all__ = (
    "GameDataset",
    "PositionDataset",
)

import json
import random
from typing import List

import chess

from chessrl import game


class GameDataset:
    """Class for holding several games (chess.Board objects) which
    provides operations to serialize/deserialize them as a JSON file.
    """

    def __init__(self, games: List[chess.Board] = None) -> None:
        """Builds a games dataset.
        Parameters:
            games: List[chess.Board]. List of board objects containing games.
        """
        self.games = []
        if games is not None:
            self.games = games

    def load(self, path: str):
        games_file = None
        with open(path, "r") as f:
            games_file = f.read()
        self.loads(games_file)

    def loads(self, string):
        games = json.loads(string)
        for item in games:
            b = game.get_new_board()
            if len(item["moves"]) > 0:
                for m in item["moves"]:
                    game.move(b, m)
                self.games.append(b)

    def save(self, path):
        dataset_existent = GameDataset()
        try:
            dataset_existent.load(path)
        except FileNotFoundError:
            pass

        union_games = dataset_existent.games + self.games

        games = [game.get_history(b) for b in union_games]

        dstr = json.dumps(games)
        dstr = dstr.replace("},", "},\n")

        with open(path, "w") as f:
            f.write(dstr)

    def append(self, other):
        """Appends a game (or another Dataset) to this one"""
        if isinstance(other, chess.Board):
            self.games.append(other)
        elif isinstance(other, GameDataset):
            self.games.extend(other.games)

    def __str__(self):
        games = [x.get_history() for x in self.games]
        return json.dumps(games)

    def __add__(self, other):
        """Appends a game (or another Dataset) to this one"""
        self.append(other)
        return self

    def __iad__(self, other):
        """Appends a game (or another Dataset) to this one"""
        return self.__add__(other)

    def __len__(self):
        return len(self.games)

    def __getitem__(self, key):
        return self.games[key]


class PositionDataset:
    """Class for holding several positions (game-state - score pairs)
    which provides operations to serialize/deserialize them as a JSON file.
    """

    def __init__(
        self,
        game_states: List[chess.Board] = None,
        scores: List[int] = None,
    ) -> None:
        """Builds a positions dataset.
        Parameters:
            game_states: List[chess.Board]. List of board objects representing
                game states.
            scores: List[int]. List of centipawn scores for each game state.
                Representing the 'goodness' of the position for the player
                about to move.
        """
        self.game_states = []
        self.game_scores = []
        if (game_states is not None) and (scores is not None):
            self.game_states = [b.copy(stack=False) for b in game_states]
            self.game_scores = scores

    def load(self, path: str):
        """Load positions from JSON of (FEN, score) entries."""
        positions_file = None
        with open(path, "r") as f:
            positions_file = f.read()
        self.loads(positions_file)

    def loads(self, string: str):
        positions = json.loads(string)
        for item in positions:
            b = game.get_new_board()
            b.set_fen(item[0])
            s = item[1]
            self.add_position(b, s)

    def save(self, path: str):
        """Save dataset of positions to JSON of (FEN, score) entries."""
        all_data = PositionDataset()
        try:
            all_data.load(path)
        except FileNotFoundError:
            pass

        all_data.append(self)

        fens = [b.fen() for b in all_data.game_states]
        scores = all_data.game_scores
        positions = [(p, s) for p, s in zip(fens, scores)]

        dstr = json.dumps(positions)
        dstr = dstr.replace("],", "],\n")

        with open(path, "w") as f:
            f.write(dstr)

    def add_position(self, board: chess.Board, score: int):
        """Adds a (game-state, score) pair to the dataset."""
        self.game_states.append(board.copy(stack=False))
        self.game_scores.append(score)

    def append(self, other):
        """Appends a game (or another Dataset) to this one"""
        if isinstance(other, PositionDataset):
            self.game_states.extend(other.game_states)
            self.game_scores.extend(other.game_scores)
        else:
            raise TypeError("Can only append another PositionDataset object.")

    def shuffle(self):
        """Shuffle the dataset in place."""
        combined = list(zip(self.game_states, self.game_scores))
        random.shuffle(combined)
        self.game_states[:], self.game_scores[:] = zip(*combined)

    def __str__(self):
        fens = [b.board_fen() for b in self.game_states]
        scores = self.game_scores
        positions = [(p, s) for p, s in zip(fens, scores)]
        return json.dumps(positions)

    def __len__(self):
        assert len(self.game_states) == len(self.game_scores)
        return len(self.game_states)

    def __getitem__(self, key):
        return (self.game_states[key], self.game_scores[key])
