__all__ = ("GameDataset",)

import json
from typing import List

import chess

from chessrl import game


class GameDataset(object):
    """
    This class holds several games and provides operations to
    serialize/deserialize them as a JSON file. Also, it takes a game and
    returns it as the expanded game.
    """

    def __init__(self, games: List[chess.Board] = None) -> None:
        """Builds a dataset.
        Parameters:
            games: List[chess.Board]. List of board objects containing games.
        """
        self.games = []
        if games is not None:
            self.games = games

    # def augment_game(self, game_base):
    #     # TODO: I think this is no longer needed given how we're ingesting training data now
    #     """Expands a game. For the N movements of a game, it creates
    #     N games with each state + the final result of the original game +
    #     the next movement (in each state).
    #     """
    #     hist = game_base.get_history()
    #     moves = hist["moves"]
    #     result = hist["result"]
    #     date = hist["date"]

    #     augmented = []

    #     g = Game(date=date)

    #     for m in moves:
    #         augmented.append({"game": g, "next_move": m, "result": result})
    #         g = g.get_copy()
    #         g.move(m)

    #     return augmented

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
