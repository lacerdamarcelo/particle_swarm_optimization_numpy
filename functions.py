import numpy as np


class SphereFunction:

    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions

    def evaluate_function(self, arr):
        return -1 * np.sum(np.square(arr))


class RandomFunction:

    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions
        self.memory = {}

    def evaluate_function(self, arr):
        arr_str = str(arr)
        if arr_str in self.memory:
            return self.memory[arr_str]
        else:
            value = -1 * ((np.random.random_sample() * 2000) - 1000)
            self.memory[arr_str] = value
            return value
