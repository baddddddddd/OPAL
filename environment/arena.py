import random
import os

import torch
import torch.multiprocessing as mp

from environment import Game
from environment.player import Player



class Arena:
    def __init__(self, max_player: Player, min_player: Player):
        self.max_player = max_player
        self.min_player = min_player
        self.game: Game = max_player.get_game()

        assert self.max_player.get_game() == self.min_player.get_game(), "Games for two players are not the same"


    def pit_with_replay(self, max_random_moves=0):
        self.game.reset_state()
        
        for i in range(random.randint(0, max_random_moves)):
            if self.game.is_game_over():
                break

            random_move = random.choice(self.game.get_moves())
            self.game.make_move(random_move)


        states = [self.game.get_state_tensor()]

        while not self.game.is_game_over():
            if self.game.is_maximizing():
                self.max_player.play()
            else:
                self.min_player.play()

            states.append(self.game.get_state_tensor())

        outcome = self.game.get_outcome()
        return states, outcome


    def pit_with_replay_worker(self, task_queue: mp.Queue, result_queue: mp.Queue, max_random_moves=0):
        while True:
            task = task_queue.get()
            if task is None:
                break

            states, outcome = self.pit_with_replay(max_random_moves=max_random_moves)
            safe_states = torch.stack([s.detach().cpu() for s in states])
            result_queue.put((safe_states.numpy(), outcome))


    def tournament_with_replay(self, game_count, max_random_moves=0, parallelized=False):
        if parallelized:
            # parallelization via multiprocessing, persistent workers
            if not mp.get_start_method(allow_none=True):
                mp.set_start_method("fork", force=True)

            NUM_WORKERS = os.cpu_count()
            task_queue = mp.Queue()
            result_queue = mp.Queue()

            workers = []
            for _ in range(NUM_WORKERS):
                p = mp.Process(target=self.pit_with_replay_worker, args=(task_queue, result_queue, max_random_moves))
                p.start()
                workers.append(p)

            # queue up games
            for _ in range(game_count):
                task_queue.put(1)

            # stop workers once games are processed
            for _ in range(NUM_WORKERS):
                task_queue.put(None)

            # collect results
            replay_states = []
            replay_outcome = []

            for _ in range(game_count):
                safe_states, outcome = result_queue.get()
                states = torch.from_numpy(safe_states) 
                replay_states.extend(states)
                replay_outcome.extend([outcome] * len(states))

            # join workers
            for p in workers:
                p.join()

            return replay_states, replay_outcome
        else:
            replay_states = []
            replay_outcome = []

            for _ in range(game_count):
                states, outcome = self.pit_with_replay(max_random_moves=max_random_moves)
                replay_states.extend(states)
                replay_outcome.extend([outcome] * len(states))

            return replay_states, replay_outcome


    def pit(self, show_game=False, max_random_moves=0):
        self.game.reset_state()

        for i in range(random.randint(0, max_random_moves)):
            if self.game.is_game_over():
                break

            random_move = random.choice(self.game.get_moves())
            self.game.make_move(random_move)

        while not self.game.is_game_over():
            if self.game.is_maximizing():
                self.max_player.play()
            else:
                self.min_player.play()

            if show_game:
                print(self.game)

        return self.game.get_outcome()


    def tournament(self, game_count, verbose=False, show_game=False, max_random_moves=0):
        max_wins, min_wins, draws = 0, 0, 0

        for i in range(1, game_count + 1):
            outcome = self.pit(show_game=show_game, max_random_moves=max_random_moves)
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
        