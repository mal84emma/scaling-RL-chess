import chess
from game import Game
from stockfish import Stockfish


class GameStockfish(Game):
    """ Represents a game agaisnt a Stockfish Agent."""

    def __init__(self, stockfish,
                 player_color=Game.WHITE,
                 board=None,
                 date=None,
                 stockfish_depth=10,
                 stockfish_elo=1320):
        super().__init__(board=board, player_color=player_color, date=date)
        if stockfish is None:
            raise ValueError('A Stockfish object or a path is needed.')

        self.stockfish = stockfish
        stockfish_color = not self.player_color

        if type(stockfish) == str:
            self.stockfish = Stockfish(stockfish_color,
                                       stockfish,
                                       search_depth=stockfish_depth,
                                       elo=stockfish_elo)
        elif type(stockfish) == Stockfish:
            self.stockfish = stockfish

    def move(self, movement):
        """ Makes a move. If it's not your turn, Stockfish will play and if
            the move is illegal, it will be ignored.
        Params:
            movement: str, Movement in UCI notation (f2f3, g8f6...)
        """
        # If stockfish moves first
        if (self.stockfish.color == Game.WHITE) and (len(self.board.move_stack) == 0):
            stockfish_move = self.stockfish.get_move(self)
            self.board.push(chess.Move.from_uci(stockfish_move))
        else:
            made_movement = super().move(movement)
            if made_movement and self.get_result() is None:
                stockfish_move = self.stockfish.get_move(self)
                self.board.push(chess.Move.from_uci(stockfish_move))

    def get_copy(self):
        return GameStockfish(board=self.board.copy(), stockfish=self.stockfish)

    def close(self):
        """ Close the connection to the engine. This needs to be done
        explicitly for each engine, otherwise the engine process will
        remain active, and the program can hang due to an async lock.
        The descturctor is not guaranteed to clean up the engine at
        the right time.
        """
        if hasattr(self, 'stockfish') and self.stockfish:
            self.stockfish.close()
