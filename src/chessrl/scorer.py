from game import Game


class Scorer(object):
    """This class represents contains the necessary methods all chess
    scorer objects must implement.
    """
    def __init__(self):
        if type(self) is Scorer:
            raise Exception('Cannot create Scorer Abstract class.')

    def score_position(self, game: Game) -> dict:  # noqa: E0602, F821
        """Evaluates the strength of a board position for the player that is
        about to take a turn.
        
        Args:
            game: An object of the Game class which describes a game position
            
        Returns:
            scores: A dictionary of scores which has the following key value 
                   pairs; {'cp': cp_score, 'rate': score_rate}
        """
        raise Exception('Abstract class.')