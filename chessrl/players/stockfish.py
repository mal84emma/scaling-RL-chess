import chess
import chess.engine
from chess.engine import SimpleEngine, Limit
import logging

import random
from chessrl.agents.player import Player
from game import Game

# Remove anoying warnings of the engine.
chess.engine.LOGGER.setLevel(logging.ERROR)


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class Stockfish(Player):
    """ AI using Stockfish to play a game of chess."""

    def __init__(self,
                 color: bool,
                 binary_path: str,
                 thinking_time=0.1,
                 search_depth=10,
                 elo=1320
                ):
        super().__init__(color)

        self.engine: SimpleEngine = SimpleEngine.popen_uci(binary_path)
        #print(list(self.engine.options))
        self.limit: Limit = Limit(time=thinking_time,depth=search_depth)
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        # page below discusses effect of Stockfish level on Elo rating
        # https://chess.stackexchange.com/questions/29860/is-there-a-list-of-approximate-elo-ratings-for-each-stockfish-level

        self.elo = elo

    def get_move(self, game: Game):

        assert game.board.turn == self.color, \
            "It's not Stockfish's turn to play."

        result = self.engine.play(game.board, self.limit)
        move = result.move.uci()

        # add a bit of stochasticity to Stockfish's move choice, select move
        # either one better or one worse than given elo choice randomly
        variations = self.engine.analyse(game.board,
                                         self.limit,
                                         multipv=50)
        move_variations = [v['pv'][0].uci() for v in variations]

        if move in move_variations:
            move_index = move_variations.index(move)
            if random.random() <= 0.5:
                increment = random.choice([-1, 1])
                move_index = clamp(move_index + increment, 0, len(move_variations)-1)
                move = move_variations[move_index]

        return move

    def close(self):
        """Close the connection to the engine. This needs to be done
        explicitly for each engine, otherwise the engine process will
        remain active, and the program can hang due to an async lock.
        The descturctor is not guaranteed to clean up the engine at
        the right time."""
        if hasattr(self, 'engine') and self.engine:
            self.engine.quit()

