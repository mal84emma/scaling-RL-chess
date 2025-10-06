"""
Module with all the necessary stuff to encode/decode game states for using them
with a neural network.

##Not anymore - Also contains a Sequence Generator for fitting a
DatasetGame to a Model.
"""

import chess
import numpy as np


def _get_pieces_one_hot(board, color=False):
    """Returns a 3D-matrix representation of the pieces for one color.
    The matrix ins constructed as follows:
        8x8 (Chess board) x 6 possible pieces. = 384.

    Parameters:
        board: Python-Chess Board. Board.
        color: Boolean, True for white, False for black
    Returns:
        mask: numpy array, 3D matrix with the pieces of the player.
    """
    mask = np.zeros((8, 8, len(chess.PIECE_TYPES)))
    for i, piece_id in enumerate(chess.PIECE_TYPES):
        mask[:, :, i] = np.array(
            board.pieces(piece_id, color).mirror().tolist()
        ).reshape(8, 8)
    # Encode blank positions
    # mask[:, :, 0] = (~np.array(mask.sum(axis=-1), dtype=bool)).astype(int)
    return mask


def _get_pieces_planes(board):
    """This method returns the matrix representation of a game turn
    (positions of the pieces of the two colors)

    Parameters:
        board: Python-Chess board
    Returns:
        current: numpy array. 3D Matrix with dimensions 14x8x8.
    """
    # TODO: castling rights, en passant, 50 move rule, player turn etc? this isn't the full game state
    # see chess programming wiki for ideas
    # https://www.chessprogramming.org/Board_Representation#FEN_Board_Representation

    # get one-hot encoding of pieces for each color
    black_pieces = _get_pieces_one_hot(board, color=False)
    white_pieces = _get_pieces_one_hot(board, color=True)
    # concatenate
    # NOTE empty squares are implied where there are no pieces of either color
    all_pieces = np.concatenate([white_pieces, black_pieces], axis=-1)

    return all_pieces


def _get_en_passant_plane(board: chess.Board):
    """Returns a matrix with the en passant square if available.

    Parameters:
        board: Python-Chess board
    Returns:
        en_passant: numpy array. 8x8 matrix with 1 in the en passant square
                    and 0 elsewhere. If no en passant is available, returns
                    a matrix of 0's.
    """
    en_passant = np.zeros((8, 8))
    if board.ep_square is not None:
        row = chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        en_passant[row, col] = 1
    return en_passant


def _get_castling_planes(board: chess.Board):
    """Returns four planes with the castling rights for each color.

    Parameters:
        board: Python-Chess board
    Returns:
        castling: numpy array. 8x8x4 matrix with 1's in all squares if the
                  corresponding color can castle that side, and 0's otherwise.
                  Planes 1, 2 are white king/queen side,
                  Planes 3, 4 are black king/queen side.
    """
    castling = np.zeros((8, 8, 4))
    if board.has_kingside_castling_rights(chess.WHITE):
        castling[:, :, 0] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling[:, :, 1] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        castling[:, :, 2] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        castling[:, :, 3] = 1

    return castling


def _get_side_to_move_plane(board: chess.Board):
    """Returns a matrix with the side to move.

    Parameters:
        board: Python-Chess board
    Returns:
        side_to_move: numpy array. 8x8x1 matrix with all 1's if white to move,
                      all 0's if black to move.
    """
    side_to_move = np.full((8, 8, 1), board.turn, dtype=int)
    return side_to_move


'''
def _get_game_history(board, T=8):
    """ Returns the matrix representation of a board history. If a game has no
    history, those turns will be considered null (represented as 0's matrices).

    Parameters:
        board: Python-Chess board (with or without moves)
        T: number of backwards steps to represent. (default 8 as in AlphaZero).
    Returns:
        history: NumPy array of dimensions 8x8x(14*T). Note that this history
        does not include the current game state, only the previous ones.
    """
    board_copy = board.copy()
    history = np.zeros((8, 8, 14 * T))

    for i in range(T):
        try:
            board_copy.pop()
        except IndexError:
            break
        history[:, :, i * 14: (i + 1) * 14] =\
            _get_current_game_state(board_copy)

    return history
'''


def get_game_state(game, flipped=False):
    """This method returns the matrix representation of a game with its
    history of moves.

    Parameters:
        game: Game. Game state.
    Returns:
        game_state: numpy array. 3D Matrix with dimensions 8x8x[14(T+1)]. Where T
        corresponds to the number of backward turns in time.
    """
    board = game.board

    pieces = _get_pieces_planes(board)
    side_to_move = _get_side_to_move_plane(board)
    castling_rights = _get_castling_planes(board)
    en_passant = _get_en_passant_plane(board)

    # NOTE this representation is missing:
    # - 50 move rule counter
    # - move repetition count
    # These can be hard-implemented by the search algorithm if needed
    game_state = np.concatenate(
        [pieces, side_to_move, castling_rights, en_passant], axis=-1
    )

    # Why flip the board?
    if flipped:
        game_state = np.rot90(game_state, k=2)
    return game_state
