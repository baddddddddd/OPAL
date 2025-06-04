from environment import Game
import torch

class TicTacToe(Game):
    def __init__(self):
        self.x_board = 0
        self.o_board = 0
        self.is_x_to_move = True
        self.history = []
        self.NUM_COLS = 3
        self.NUM_ROWS = 3

        self.lines = [
            0b000000111,
            0b000111000,
            0b111000000,
            0b001001001,
            0b010010010,
            0b100100100,
            0b100010001,
            0b001010100,
        ]


    def reset_state(self):
        self.x_board = 0
        self.o_board = 0
        self.is_x_to_move = True
        self.history = []


    def get_moves(self):
        occupied = self.x_board | self.o_board
        moves = []

        for index in range(9):
            bitmask = 1 << index

            if not (occupied & bitmask):
                moves.append(index)

        # import random
        # random.shuffle(moves)
        return moves


    def make_move(self, move):
        bitmask = 1 << move
        
        if self.is_x_to_move:
            self.x_board |= bitmask
        else:
            self.o_board |= bitmask

        self.is_x_to_move = not self.is_x_to_move
        self.history.append(move)


    def undo_move(self):
        if len(self.history) == 0:
            return

        last_move = self.history.pop()
        bitmask = 1 << last_move

        self.is_x_to_move = not self.is_x_to_move

        if self.is_x_to_move:
            self.x_board &= ~bitmask
        else:
            self.o_board &= ~bitmask


    def evaluate(self):
        outcome = self.get_outcome()
        if outcome is None:
            return 0

        return outcome * (10 - len(self.history))        


    def get_outcome(self):
        for line in self.lines:
            if (self.x_board & line) == line:
                return 1 
            if (self.o_board & line) == line:
                return -1

        if len(self.history) == 9:
            return 0

        return None


    def is_game_over(self):
        return self.get_outcome() is not None


    def is_maximizing(self):
        return self.is_x_to_move


    def __str__(self):
        rows = []
        for i in range(3):
            row = ""
            for j in range(3):
                index = i * 3 + j
                bitmask = 1 << index

                if self.x_board & bitmask:
                    row += 'X'
                elif self.o_board & bitmask:
                    row += 'O'
                else:
                    row += '.'

            rows.append(row)

        return '\n'.join(rows) + "\n"


    def get_board_tensor(self, board):
        rows = []
        for i in range(self.NUM_ROWS):
            row = []
            for j in range(self.NUM_COLS):
                index = (i * self.NUM_COLS) + j
                bitmask = 1 << index

                row.append(1 if (board & bitmask) else 0)

            rows.append(torch.tensor(row, dtype=torch.float32))

        return torch.stack(rows)


    def get_state_tensor(self):
        x_tensor = self.get_board_tensor(self.x_board)
        o_tensor = self.get_board_tensor(self.o_board)
        turn_tensor = torch.ones((self.NUM_ROWS, self.NUM_COLS), dtype=torch.float32) if self.is_maximizing() else torch.zeros((self.NUM_ROWS, self.NUM_COLS), dtype=torch.float32)
        
        return torch.stack([x_tensor, o_tensor, turn_tensor])
