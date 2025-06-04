from environment import Game
from environment.player import Player

class Arena:
    def __init__(self, max_player: Player, min_player: Player):
        self.max_player = max_player
        self.min_player = min_player
        self.game: Game = max_player.get_game()

        assert self.max_player.get_game() == self.min_player.get_game(), "Games for two players are not the same"


    def pit(self, show_game=False):
        self.game.reset_state()

        while not self.game.is_game_over():
            if self.game.is_maximizing():
                self.max_player.play()
            else:
                self.min_player.play()

            if show_game:
                print(self.game)

        return self.game.get_outcome()


    def tournament(self, game_count, verbose=False, show_game=False):
        max_wins, min_wins, draws = 0, 0, 0

        for i in range(1, game_count + 1):
            outcome = self.pit(show_game=show_game)
            if outcome == 1:
                max_wins += 1
                message = "Max Player Wins"
            elif outcome == -1:
                min_wins += 1
                message = "Min Player Wins"
            else:
                draws += 1
                message = "Draw"

            if verbose:
                print(f"Game {i}: {message}")

        return max_wins, min_wins, draws
        