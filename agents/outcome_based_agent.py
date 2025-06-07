import torch
from torch import nn

from agents import Agent
from environment import Game


# 0 - maximizing win rate
# 1 - draw rate
# 2 - minimizing win rate

class OutcomeBasedAgent(Agent):
    def __init__(self, model: nn.Module, device, game: Game):
        super().__init__(game)

        self.model = model
        self.device = device


    def find_move(self, temperature=0.0):
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
            probabilities = torch.softmax(beliefs, dim=1)

            # E = P(Win) + 0.5 * P(Draw), i know, too simple
            if self.game.is_maximizing():
                expected_scores = probabilities[:, 0] + 0.5 * probabilities[:, 1]
            else:
                expected_scores = probabilities[:, 2] + 0.5 * probabilities[:, 1]

            move_index = torch.argmax(expected_scores).item()
            return moves[move_index]


    def play(self):
        move = self.find_move()
        self.game.make_move(move)