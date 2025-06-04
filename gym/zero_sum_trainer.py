import os
import re

import torch
from torch import nn

from environment import Game, Arena, Player

class ModelPlayer(Player):
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


class ZeroSumTrainer:
    def __init__(self, model: nn.Module, game: Game):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model = model.to(self.device)

        self.game = game
        self.max_player = ModelPlayer(self.model, self.device, self.game)
        self.min_player = ModelPlayer(self.model, self.device, self.game)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)


    def step(self, batch_size, max_random_moves=0):
        arena = Arena(self.max_player, self.min_player)
        states, outcome = arena.tournament_with_replay(game_count=batch_size, max_random_moves=max_random_moves)

        self.model.train()
        states = torch.stack(states).to(self.device)
        outcome = torch.tensor(outcome, dtype=torch.float32, device=self.device).unsqueeze(1)
        prediction = self.model(states)

        loss = self.loss_fn(prediction, outcome)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss


    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


    def save_checkpoint(self, filename):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, filename)


    def get_latest_checkpoint(self, folder):
        checkpoints = [int(re.search(r"step_(\d+)\.pth", f)[1]) for f in os.listdir(folder) if re.match(r"step_(\d+)\.pth", f)]
        latest = max(checkpoints, default=0)
        return latest


    def train(self, step_count, batch_size, max_random_moves=0, log_interval=1, checkpoint_interval=None, checkpoint_dir=None, resume_checkpoint=False):
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if resume_checkpoint:
            if checkpoint_dir is None:
                raise Exception("Must also provide checkpoint_dir")

            latest = self.get_latest_checkpoint(checkpoint_dir)
            start_step = latest + 1

            filename = f"{checkpoint_dir}/step_{latest:06d}.pth"
            if os.path.exists(filename):
                self.load_checkpoint(filename)
                print(f"Resuming from {filename}")
            else:
                filename = f"{checkpoint_dir}/step_{0:06d}.pth"
                self.save_checkpoint(filename)
                print(f"Saved checkpoint to {filename}")


        if checkpoint_interval:
            if checkpoint_dir is None:
                raise Exception("Must also provide checkpoint_dir")


        for i in range(start_step, step_count + 1):
            loss = self.step(batch_size=batch_size, max_random_moves=max_random_moves)

            if i % log_interval == 0:
                print(f"Step {i:>6d}:    Loss={loss.item():>.8f}")

            if checkpoint_interval is not None and i % checkpoint_interval == 0:
                filename = f"{checkpoint_dir}/step_{i:06d}.pth"
                self.save_checkpoint(filename)
                print(f"Saved checkpoint to {filename}")
                
