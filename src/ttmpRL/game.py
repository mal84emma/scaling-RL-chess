from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ttmpRL.players import Player

from datetime import datetime
from io import BytesIO

import cairosvg
import chess
import chess.svg
import numpy as np
from chess import Board
from PIL import Image


# TODO: Is this class redundant with chess.Board? all of the methods could be
# implemented as pure functions operating on a chess.Board object
class Game(object):
    NULL_MOVE = "00000"

    def __init__(
        self,
        white_player: Player = None,
        black_player: Player = None,
        board: Board = None,
        date=None,
    ):
        self.white_player = white_player
        self.black_player = black_player

        if board is None:
            self.board = Board()
        else:
            self.board = board

        self.date = date
        if self.date is None:
            self.date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    def next_move(self):
        """Retrieves and makes the move of the player who's turn it is."""
        if self.turn == chess.WHITE:
            player_to_move = self.white_player
        else:
            player_to_move = self.black_player

        movement = player_to_move.get_move(self)

        self.move(movement)

    def move(self, movement):
        """Make a specified move.

        Params:
            movement: str, Movement in UCI notation (e.g. f2f3 or g8f6).
        """
        player_color = "White" if self.turn == chess.WHITE else "Black"
        assert movement in self.get_legal_moves(), (
            f"{player_color} player tried to make an illegal move: {movement}"
        )

        self.board.push(chess.Move.from_uci(movement))

    def get_legal_moves(self, final_states=False):
        """Gets a list of legal moves in the current turn.
        Parameters:
            final_states: bool. Whether copies of the board after executing
            each legal movement are returned.
        """
        moves = [m.uci() for m in self.board.legal_moves]
        if final_states:
            states = []
            for m in moves:
                gi = self.get_copy()
                gi.move(m)
                states.append(gi)
            moves = (moves, states)
        return moves

    def get_history(self):
        moves = [x.uci() for x in self.board.move_stack]
        res = self.get_result()
        return {"result": res, "moves": moves, "date": self.date}

    def get_fen(self):
        return self.board.board_fen()

    def set_fen(self, fen):
        self.board.set_board_fen(fen)

    @property
    def turn(self):
        """Returns whether is white turn."""
        return self.board.turn

    def get_copy(self):
        return Game(board=self.board.copy())

    def reset(self):
        self.board.reset()

    def close(self):
        """Closes any resources used by the game (e.g. engines)."""
        self.white_player.close()
        self.black_player.close()

    def get_result(self):
        """Returns the result of the game for the white pieces. None if the
        game is not over. This method checks if the game ends in a draw due
        to the fifty-move rule. Threefold is not checked because it can
        be too slow.
        """
        result = None
        if self.board.can_claim_fifty_moves():
            result = 0
        elif self.board.is_game_over():
            r = self.board.result()
            if r == "1-0":
                result = 1  # Whites win
            elif r == "0-1":
                result = -1  # Whites dont win
            else:
                result = 0  # Draw
        return result

    def __len__(self):
        return len(self.board.move_stack)

    def plot_board(
        self, return_img=False, show_moves=True, orientation=chess.WHITE, save_path=None
    ):
        """Plots the current state of the board. This is useful for debug/log
        purposes while working outside a notebook

        Parameters:
            save_path: str, where to save the image. None if you want to plot
            on the screen only
        """

        if show_moves:
            prev_moves = [x.uci() for x in self.board.move_stack[-2:]]
            arrows = [chess.svg.Arrow.from_pgn(x[:4]) for x in prev_moves]
        else:
            arrows = []

        svg = chess.svg.board(self.board, orientation=orientation, arrows=arrows)

        out = BytesIO()
        cairosvg.svg2png(svg, write_to=out)
        image = Image.open(out)

        if return_img:
            return np.asarray(image)
        if save_path is not None:
            image.save(save_path)
