import pickle
import os

from environment import Game

class Connect4(Game):
    # constants
    NUM_COLS = 7
    NUM_ROWS = 6
    NUM_TOTAL = NUM_COLS * NUM_ROWS
    NUM_CONNECT = 4

    TABLES_FOLDER = "tables"
    MOVES_PKL = f"{TABLES_FOLDER}/moves.pkl"
    LINES_PKL = f"{TABLES_FOLDER}/lines.pkl"

    moves = {}
    lines = []

    def __init__(self):
        # initial state
        self.red_board = 0
        self.yellow_board = 0
        self.is_red_to_move = True
        self.history = []

        # build hash tables
        if not Connect4.load_moves_table():
            Connect4.build_move_map()

        if not Connect4.load_lines_table():
            Connect4.build_winning_lines()


    def reset_state(self):
        self.red_board = 0
        self.yellow_board = 0
        self.is_red_to_move = True
        self.history = []


    @staticmethod
    def load_pickle(filepath):        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)

        return None


    @staticmethod
    def save_pickle(filepath, obj):
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load_moves_table():
        if Connect4.moves:
            return True

        os.makedirs(Connect4.TABLES_FOLDER, exist_ok=True)
        Connect4.moves = Connect4.load_pickle(Connect4.MOVES_PKL) or {}
        return len(Connect4.moves) > 0


    @staticmethod
    def load_lines_table():
        if Connect4.lines:
            return True

        os.makedirs(Connect4.TABLES_FOLDER, exist_ok=True)
        Connect4.lines = Connect4.load_pickle(Connect4.LINES_PKL) or []
        return len(Connect4.lines) > 0
        

    @staticmethod
    def build_move_map():
        bitmask = 1 << (Connect4.NUM_COLS * (Connect4.NUM_ROWS - 1))
        magic = 0

        col_moves = [list() for _ in range(Connect4.NUM_COLS)]

        for i in range(Connect4.NUM_ROWS + 1):
            move = bitmask >> (Connect4.NUM_COLS * i)
            if i == Connect4.NUM_ROWS:
                move = 0

            for j in range(Connect4.NUM_COLS):
                shifted_move = move << j
                shifted_magic = magic << j

                col_moves[j].append((shifted_magic, shifted_move))

            magic |= move


        def build_magics(depth, magic, move):
            if depth == Connect4.NUM_COLS:
                Connect4.moves[magic] = move
                return

            for row in range(Connect4.NUM_ROWS + 1):
                submagic, submove = col_moves[depth][row]
                next_magic = magic | submagic
                next_move = move | submove
                build_magics(depth + 1, next_magic, next_move)

        build_magics(0, 0, 0)

        Connect4.save_pickle(Connect4.MOVES_PKL, Connect4.moves)


    @staticmethod
    def build_winning_lines():
        def shift_line(line, end_row, end_col):
            for i in range(end_row):
                for j in range(end_col):
                    vertical_shift = i * Connect4.NUM_COLS
                    horizontal_shift = j
                    winning_line = (line << vertical_shift) << horizontal_shift
                    Connect4.lines.append(winning_line)

        # horizontal
        line = 0b1111
        shift_line(line, Connect4.NUM_ROWS, Connect4.NUM_COLS - Connect4.NUM_CONNECT + 1)

        # vertical
        bitmask = 1
        line = 0
        for i in range(Connect4.NUM_CONNECT):
            line |= bitmask << (i * Connect4.NUM_COLS)

        shift_line(line, Connect4.NUM_ROWS - Connect4.NUM_CONNECT + 1, Connect4.NUM_COLS)

        # neg diagonal
        bitmask = 1
        line = 0
        for i in range(Connect4.NUM_CONNECT):
            line |= (bitmask << i) << (i * Connect4.NUM_COLS)

        shift_line(line, Connect4.NUM_ROWS - Connect4.NUM_CONNECT + 1, Connect4.NUM_COLS - Connect4.NUM_CONNECT + 1)

        # pos diagonal
        bitmask = 1
        line = 0
        for i in range(Connect4.NUM_CONNECT):
            line |= (bitmask << (Connect4.NUM_CONNECT - i - 1))  << (i * Connect4.NUM_COLS)

        shift_line(line, Connect4.NUM_ROWS - Connect4.NUM_CONNECT + 1, Connect4.NUM_COLS - Connect4.NUM_CONNECT + 1)

        Connect4.save_pickle(Connect4.LINES_PKL, Connect4.lines)


    def get_moves(self):
        occupied = self.red_board | self.yellow_board

        moves = self.moves[occupied]
        move_list = []

        while moves:
            move = moves & -moves
            moves &= moves - 1
            move_list.append(move)

        return move_list


    def make_move(self, move):
        if self.is_red_to_move:
            self.red_board |= move
        else:
            self.yellow_board |= move

        self.is_red_to_move = not self.is_red_to_move
        self.history.append(move)


    def undo_move(self):
        if len(self.history) == 0:
            return

        last_move = self.history.pop()

        self.is_red_to_move = not self.is_red_to_move

        if self.is_red_to_move:
            self.red_board &= ~last_move
        else:
            self.yellow_board &= ~last_move


    def evaluate(self):
        outcome = self.get_outcome()
        if outcome is None:
            return 0

        return outcome * (self.NUM_TOTAL - len(self.history) + 1)        

        # score = 0

        # # Center control bonus
        # center_col = [i * self.NUM_COLS + self.NUM_COLS // 2 for i in range(self.NUM_ROWS)]
        # center_mask = sum(1 << idx for idx in center_col)
        # red_center_count = bin(self.red_board & center_mask).count('1')
        # yellow_center_count = bin(self.yellow_board & center_mask).count('1')
        # score += 3 * (red_center_count - yellow_center_count)

        # # Heuristic scoring based on open lines
        # for line in self.lines:
        #     red = self.red_board & line
        #     yellow = self.yellow_board & line

        #     if red and yellow:
        #         continue  # line is blocked

        #     count = bin(red | yellow).count('1')
        #     if count == 0:
        #         continue  # empty line, not urgent

        #     if red:
        #         num = bin(red).count('1')
        #         if num == 4:
        #             return 1000000
        #         elif num == 3:
        #             score += 100
        #         elif num == 2:
        #             score += 10
        #     elif yellow:
        #         num = bin(yellow).count('1')
        #         if num == 4:
        #             return -1000000
        #         elif num == 3:
        #             score -= 100
        #         elif num == 2:
        #             score -= 10

        # return score



    def get_outcome(self):
        if len(self.history) == self.NUM_TOTAL:
            return 0

        for line in self.lines:
            if (self.red_board & line) == line:
                return 1 
            if (self.yellow_board & line) == line:
                return -1

        return None


    def is_game_over(self):
        return self.get_outcome() is not None


    def is_maximizing(self):
        return self.is_red_to_move


    def __str__(self):
        rows = []
        for i in range(self.NUM_ROWS):
            row = ""
            for j in range(self.NUM_COLS):
                index = i * self.NUM_COLS + j
                bitmask = 1 << index

                if self.red_board & bitmask:
                    row += 'R'
                elif self.yellow_board & bitmask:
                    row += 'Y'
                else:
                    row += '.'

            rows.append(row)

        return '\n'.join(rows) + "\n"


Connect4.load_moves_table()
Connect4.load_lines_table()