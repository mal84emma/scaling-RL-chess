import chess
import chess.engine
import logging

import random
from player import Player
from game import Game

# Remove anoying warnings of the engine.
chess.engine.LOGGER.setLevel(logging.ERROR)


class Stockfish(Player):
    """ AI using Stockfish to play a game of chess."""

    def __init__(self,
                 color: bool,
                 binary_path: str,
                 thinking_time=0.01,
                 search_depth=10,
                 rand_depth=False,
                 move_quality=1.0,
                 training_mode=False
                ):
        super().__init__(color)
        self.engine = chess.engine.SimpleEngine.popen_uci(binary_path)

        self.thinking_time = thinking_time
        self.search_depth = search_depth
        self.rand_depth = rand_depth
        self.move_quality = move_quality
        self.training_mode = training_mode

    def best_move(self, game: Game, first_move: bool = False):  # noqa: E0602, F821
        # Page 77 of
        # http://web.ist.utl.pt/diogo.ferreira/papers/ferreira13impact.pdf
        # gives some study about the relation of search depth vs ELO.

        # this feature is alright, but depth doesn't have much affect on move choice
        if self.rand_depth and self.search_depth > 1:
            depth = random.randint(1, self.search_depth+1)
        else:
            depth = self.search_depth

        # legacy - only returns best move within compute limit
        # result = self.engine.play(game.board,
        #                           #chess.engine.Limit(time=self.thinking_time)
        #                           chess.engine.Limit(depth=depth)
        #                           )
        # return result.move.uci()

        # to introduce stochasticity to how stockfish plays, we analyse
        # the position and generate the best 20 moves, from which we
        # pick according to a distribution (depending on mode)

        variations = self.engine.analyse(game.board,
                                         chess.engine.Limit(depth=depth),
                                         multipv=20)

        # for training data generation, skew randomness to good moves
        if self.training_mode:
            if first_move: # add more entropy to first move
                var_dist = [0,1,2,3,4,5]
            else:
                var_dist = [0]*10 + [1]*5 + [2]*3 + [3]*1 + [4]*1

            var_num = random.choice(var_dist)

        else:
            var_num = int(len(variations)*(1-self.move_quality))
            # move_quality needs to be a fractional, otherwise when there are few
            # variations it always picks the worst move and the game stalls

        var_num = min(var_num, len(variations)-1)

        return variations[var_num]['pv'][0].uci()

    def kill(self):
        self.engine.quit()
