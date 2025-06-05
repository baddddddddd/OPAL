import random

from environment import Game


class StochasticMinimax:
    def __init__(self, game: Game):
        self.game = game


    def play(self, depth: int):
        is_maximizing = self.game.is_maximizing()

        moves = self.game.get_moves()
        random.shuffle(moves)
        
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        if is_maximizing:
            value = float('-inf')
            for move in moves:
                self.game.make_move(move)
                cur = self.minimax(depth - 1, alpha, beta, not is_maximizing)
                self.game.undo_move()

                if cur > value:
                    value = cur
                    best_move = move

                alpha = max(alpha, value)

        else:
            value = float('inf')
            for move in moves:
                self.game.make_move(move)
                cur = self.minimax(depth - 1, alpha, beta, not is_maximizing)                
                self.game.undo_move()

                if cur < value:
                    value = cur
                    best_move = move

                beta = min(beta, value)

        return best_move


    def minimax(self, depth: int, alpha: float, beta: float, is_maximizing: bool):
        if depth == 0 or self.game.is_game_over():
            return self.game.evaluate()

        moves = self.game.get_moves()
        random.shuffle(moves)

        if is_maximizing:
            value = float('-inf')
            for move in moves:
                self.game.make_move(move)
                value = max(value, self.minimax(depth - 1, alpha, beta, not is_maximizing))
                self.game.undo_move()

                if value >= beta:
                    break

                alpha = max(alpha, value)

        else:
            value = float('inf')
            for move in moves:
                self.game.make_move(move)
                value = min(value, self.minimax(depth - 1, alpha, beta, not is_maximizing))
                self.game.undo_move()

                if value <= alpha:
                    break

                beta = min(beta, value)

        return value
