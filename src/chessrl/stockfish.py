import chess
import chess.engine
import logging

import random
from player import Player
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
                 thinking_time=0.01,
                 search_depth=10,
                 elo=1320
                ):
        super().__init__(color)

        self.engine = chess.engine.SimpleEngine.popen_uci(binary_path)
        #print(list(self.engine.options))
        self.engine.configure({"UCI_Elo": elo})
        # page below discusses effect of Stockfish level on Elo rating
        # https://chess.stackexchange.com/questions/29860/is-there-a-list-of-approximate-elo-ratings-for-each-stockfish-level

        self.thinking_time = thinking_time
        self.search_depth = search_depth
        self.elo = elo

    def get_move(self, game: Game):

        result = self.engine.play(game.board,
                                  chess.engine.Limit(time=self.thinking_time,
                                                     depth=self.search_depth)
                                  )
        move = result.move.uci()

        # add a bit of stochasticity to Stockfish's move choice, select move
        # either one better or one worse than given elo choice randomly
        variations = self.engine.analyse(game.board,
                                         chess.engine.Limit(time=self.thinking_time,
                                                     depth=self.search_depth),
                                         multipv=50)
        move_variations = [v['pv'][0].uci() for v in variations]

        if move in move_variations:
            move_index = move_variations.index(move)
            if random.random() <= 0.5:
                increment = random.choice([-1, 1])
                move_index = clamp(move_index + increment, 0, len(move_variations)-1)
                print('old move:', move, 'new move:', move_variations[move_index])
                move = move_variations[move_index]

        return move

    def kill(self):
        self.engine.quit()
