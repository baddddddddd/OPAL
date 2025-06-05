import torch
from torch import nn

from agents import Agent
from environment import Game

class ZeroSumAgent(Agent):
    def __init__(self, model: nn.Module, device, game: Game):
        super().__init__(game)

        self.model = model
        self.device = device


    def find_move(self):
        self.model.eval()

        with torch.no_grad():
            moves = self.game.get_moves()
            lookahead = []
            for move in moves:
                self.game.make_move(move)
                lookahead.append(self.game.get_state_tensor())
                self.game.undo_move()

            lookahead = torch.stack(lookahead).to(self.device)
            beliefs = self.model(lookahead)

            if not self.game.is_maximizing():
                beliefs *= -1

            # probabilities = torch.softmax(beliefs, dim=0).flatten()
            # move_index = torch.multinomial(probabilities, num_samples=1).item()

            move_index = torch.argmax(beliefs).item()
            return moves[move_index]

    def play(self):
        move = self.find_move()
        self.game.make_move(move)