class Game:
    def __init__(self):
        pass

    def reset_state(self):
        raise NotImplementedError("reset_state() is not implemented")

    def get_moves(self):
        raise NotImplementedError("get_moves() is not implemented")

    def make_move(self, move):
        raise NotImplementedError("make_move() is not implemented")

    def undo_move(self):
        raise NotImplementedError("undo_move() is not implemented")

    def get_outcome(self):
        raise NotImplementedError("get_outcome() is not implemented")

    def is_game_over(self):
        raise NotImplementedError("is_game_over() is not implemented")

    def is_maximizing(self):
        raise NotImplementedError("is_maximizing() is not implemented")

    def __str__(self):
        raise NotImplementedError("__str__() is not implemented")

    # optional implementations
    def evaluate(self):
        raise NotImplementedError("evaluate() is not implemented")

    def get_state_tensor(self):
        raise NotImplementedError("get_state_tensor() is not implemented")
