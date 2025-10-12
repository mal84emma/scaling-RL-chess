"""Defines chess related utility functions."""

__all__ = (
    "UCIMove",
    "get_uci_labels",
)

import chessrl


class UCIMove(str):
    """Class representing a UCI move."""

    def __new__(cls, move: str):
        if move in chessrl.UCI_MOVES:
            return str.__new__(cls, move)
        else:
            raise ValueError(f"Invalid UCI move: {move}")


def get_uci_labels():
    """Returns a list of possible moves in UCI format (including
    promotions).
    Source:
        https://github.com/Zeta36/chess-alpha-zero/blob/
        master/src/chess_zero/config.py#L88
    """
    labels_array = []
    letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8"]
    promoted_to = ["q", "r", "b", "n"]

    for l1 in range(8):
        for n1 in range(8):
            destinations = (
                [(t, n1) for t in range(8)]
                + [(l1, t) for t in range(8)]
                + [(l1 + t, n1 + t) for t in range(-7, 8)]
                + [(l1 + t, n1 - t) for t in range(-7, 8)]
                + [
                    (l1 + a, n1 + b)
                    for (a, b) in [
                        (-2, -1),
                        (-1, -2),
                        (-2, 1),
                        (1, -2),
                        (2, -1),
                        (-1, 2),
                        (2, 1),
                        (1, 2),
                    ]
                ]
            )

            for l2, n2 in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):  # noqa: E501
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]  # noqa: E501
                    labels_array.append(move)

    for l1 in range(8):
        letter = letters[l1]
        for p in promoted_to:
            labels_array.append(letter + "2" + letter + "1" + p)
            labels_array.append(letter + "7" + letter + "8" + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(letter + "2" + l_l + "1" + p)
                labels_array.append(letter + "7" + l_l + "8" + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(letter + "2" + l_r + "1" + p)
                labels_array.append(letter + "7" + l_r + "8" + p)

    return labels_array
