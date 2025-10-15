__all__ = ("game", "model", "players", "scorer")

# TODO: think about what the __init__ files should look like
# do I want to expose everything at the top level?
# How I include things in __all__ and import statements
# also affects the intellisense I get in the scripts
from . import game, model, players, scorer, utils
from .dataset import *
from .game import *
from .model import *
from .players import *
from .scorer import *
from .utils import *
from .utils import get_uci_labels

NULL_MOVE = "00000"
UCI_MOVES = get_uci_labels()
