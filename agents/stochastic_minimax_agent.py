from search import StochasticMinimax
from agents import Agent

class StochasticMinimaxAgent(Agent):
    def __init__(self, game, max_depth):
        super().__init__(game)

        self.depth = max_depth
        self.search = StochasticMinimax(self.game)
        

    def play(self):
        move = self.search.play(self.depth)
        self.game.make_move(move)
    