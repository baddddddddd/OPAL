import os
import re

import torch
from torch import nn

from environment import Game, Arena
from agents import OutcomeBasedAgent


class OutcomeBasedTrainer:
    def __init__(self, model: nn.Module, game: Game, parallelized=False):
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model = model.to(self.device)
        self.parallelized = parallelized
        if self.parallelized:
            self.model.share_memory()

        self.game = game
        self.max_player = OutcomeBasedAgent(self.model, self.device, self.game)
        self.min_player = OutcomeBasedAgent(self.model, self.device, self.game)

        # self.loss_fn = nn.HuberLoss()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)


    def step(self, batch_size, max_random_moves=0):
        arena = Arena(self.max_player, self.min_player)
        replays = arena.tournament_with_replay(game_count=batch_size, max_random_moves=max_random_moves, parallelized=self.parallelized)

        MAX_WIN_CLASS = 0
        DRAW_CLASS = 1
        MIN_WIN_CLASS = 2

        data = []
        label = []
        for (replay_states, replay_outcome) in replays:
            if replay_outcome == 1:
                outcome_class = MAX_WIN_CLASS
            elif replay_outcome == -1:
                outcome_class = MIN_WIN_CLASS
            else:
                outcome_class = DRAW_CLASS

            data.extend(replay_states)
            label.extend([outcome_class] * len(replay_states))


        self.model.train()
        states = torch.stack(data).to(self.device)
        outcome = torch.tensor(label, dtype=torch.long, device=self.device)
        prediction = self.model(states)

        loss = self.loss_fn(prediction, outcome)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
                
