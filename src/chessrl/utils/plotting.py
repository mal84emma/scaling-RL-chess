"""Helper functions for plotting chess boards using matplotlib."""

__all__ = (
    "init_board_image",
    "update_board_image",
)

import chess
import matplotlib.image
import matplotlib.pyplot as plt

import chessrl.game as game
from chessrl.scorer import StockfishScorer


def init_board_image(
    board: chess.Board,
    agent_color: chess.WHITE | chess.BLACK,
    delay: float,
):
    """Initialise matplotlib image, axis, and figure for plotting
    chess game."""
    fig, ax = plt.subplots(1, 1)
    img = game.plot_board(
        board,
        return_img=True,
        show_moves=False,
        orientation=agent_color,
    )
    im = ax.imshow(img)
    ax.axis("off")
    agent_color_name = chess.COLOR_NAMES[agent_color]
    ax.set_title(f"({agent_color_name}) ...")
    plt.tight_layout()
    plt.pause(delay)

    return im, ax, fig


def update_board_image(
    board: chess.Board,
    agent_color: chess.WHITE | chess.BLACK,
    im: matplotlib.image.AxesImage,
    ax: plt.Axes,
    fig: plt.Figure,
    scorer: StockfishScorer,
    delay: float,
):
    """Update matplotlib image, axis, and figure with new game state."""
    agent_color_name = chess.COLOR_NAMES[agent_color]
    im.set_data(
        game.plot_board(
            board, return_img=True, show_moves=True, orientation=agent_color
        )
    )
    scores = scorer.score_position(board, cp_only=False)
    if board.turn == agent_color:
        ax.set_title(
            f"({agent_color_name}) CP: {scores['cp']}, rate: {scores['rate'] * 100:.1f}%"
        )
    fig.canvas.draw_idle()
    plt.pause(delay)
