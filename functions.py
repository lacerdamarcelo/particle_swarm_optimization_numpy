import numpy as np


class SphereFunction:

    def __init__(self, num_dimensions):
        self.num_dimensions = num_dimensions

    def evaluate_function(self, arr):
        return np.sum(np.square(arr))
