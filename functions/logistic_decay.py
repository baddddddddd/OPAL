import math

def logistic_decay(x, a, b):
    return 1 / (1 + math.exp(a * (x - b)))