"""Functions to play a chess game."""

import json
import random
from datetime import datetime
from io import BytesIO

import cairosvg
import chess
import chess.svg
import numpy as np
from PIL import Image

from .players import Player
from .utils import UCIMove


def get_new_board() -> chess.Board:
    """Returns a new initialised chess board."""
    return chess.Board()


def get_game_len(board: chess.Board) -> int:
    """Returns number of moves played in game."""
    return len(board.move_stack)


def get_legal_moves(board: chess.Board, final_states=False) -> list:
    """Gets a list of legal moves in the current turn.
    Parameters:
        final_states: bool. Whether copies of the board after executing
        each legal movement are returned.
    """
    moves = [m.uci() for m in board.legal_moves]
    if final_states:
        states = []
        for m in moves:
            tmp_board = get_board_copy(board)
            move(tmp_board, m)
            states.append(tmp_board)
        moves = (moves, states)
    return moves


def next_move(
    board: chess.Board,
    white_player: Player,
    black_player: Player,
    make_move=True,
) -> UCIMove:
    """Retrieves and makes the move of the player who's turn it is."""
    if board.turn == chess.WHITE:
        player_to_move = white_player
    else:
        player_to_move = black_player

    movement = player_to_move.get_move(board)

    if make_move:
        move(board, movement)

    return movement


def move(board: chess.Board, movement: UCIMove) -> None:
    """Make a specified move (inplace).

    Params:
        board: chess.Board, Current board position.
        movement: UCIMove, Movement in UCI notation (e.g. f2f3 or g8f6).
    """
    player_color = chess.COLOR_NAMES[board.turn]
    assert movement in get_legal_moves(board), (
        f"{player_color} player tried to make an illegal move: {movement}"
    )

    board.push(chess.Move.from_uci(movement))

    return


def get_history(board: chess.Board) -> dict:
    moves = [x.uci() for x in board.move_stack]
    res = get_result(board)
    date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    return {"result": res, "moves": moves, "date": date}


def get_board_copy(board: chess.Board) -> chess.Board:
    return board.copy()


def reset_board(board: chess.Board) -> None:
    board.reset()
    return


def set_opening_position(board: chess.Board, book: str) -> None:
    """Sets the board to a random opening position from the given opening book."""
    with open(book, "r") as f:
        openings = json.load(f)
    openings = [x for x in openings if ("moves" in x.keys()) and (len(x["moves"]) > 0)]

    opening = random.choice(openings)
    for movement in opening["moves"]:
        move(board, UCIMove(movement))
    return


def get_result(board: chess.Board) -> int:
    """Returns the result of the game for the white pieces. None if the
    game is not over. This method checks if the game ends in a draw due
    to the fifty-move rule. Threefold is not checked because it can
    be too slow.
    """
    result = None
    if board.can_claim_fifty_moves():
        result = 0
    elif board.is_game_over():
        r = board.result()
        if r == "1-0":
            result = 1  # Whites win
        elif r == "0-1":
            result = -1  # Whites dont win
        else:
            result = 0  # Draw
    return result


def plot_board(
    board: chess.Board,
    return_img=False,
    show_moves=True,
    orientation=chess.WHITE,
    save_path=None,
) -> None:
    """Plots the current state of the board. This is useful for debug/log
    purposes while working outside a notebook

    Parameters:
        save_path: str, where to save the image. None if you want to plot
        on the screen only
    """

    if show_moves:
        prev_moves = [x.uci() for x in board.move_stack[-2:]]
        arrows = [chess.svg.Arrow.from_pgn(x[:4]) for x in prev_moves]
    else:
        arrows = []

    svg = chess.svg.board(board, orientation=orientation, arrows=arrows)

    out = BytesIO()
    cairosvg.svg2png(svg, write_to=out)
    image = Image.open(out)

    if return_img:
        return np.asarray(image)
    if save_path is not None:
        image.save(save_path)

    return
