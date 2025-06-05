from search import Minimax
from agents import Agent

class MinimaxAgent(Agent):
    def __init__(self, game, max_depth):
        super().__init__(game)

        self.depth = max_depth
        self.search = Minimax(self.game)
        

    def play(self):
        move = self.search.play(self.depth)
        self.game.make_move(move)
    