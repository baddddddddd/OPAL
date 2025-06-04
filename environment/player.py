from environment import Game

class Player:
    def __init__(self, game: Game):
        self.game = game

    def get_game(self):
        return self.game

    def play(self, **kwargs):
        pass

