from typing import Tuple, List, Any
import numpy as np


class TupleFitness:
    vector_fitness: np.array
    size: int

    def __init__(self, vector_fitness: np.array, size: int):
        self.vector_fitness = vector_fitness
        self.size = size

    def distance(self, other):
        return np.linalg.norm(self.vector_fitness-other.vector_fitness, ord=2)

    # if none of the two solution dominates the other one.
    def __eq__(self, other):
        return not(self < other) and not(self > other)

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

    def __lt__(self, other):
        return (self.vector_fitness <= other.vector_fitness).all() \
               and (self.vector_fitness < other.vector_fitness).any()

    def __gt__(self, other):
        return (self.vector_fitness >= other.vector_fitness).all() \
               and (self.vector_fitness > other.vector_fitness).any()

    def __str__(self):
        return str(self.vector_fitness)

    def __mul__(self, other):
        return TupleFitness(other*self.vector_fitness, self.size)





