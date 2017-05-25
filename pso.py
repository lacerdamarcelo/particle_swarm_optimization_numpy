import math
import numpy as np
from functions import SphereFunction


class PSO:

    def __init__(self, initial_inertia_factor, final_inertia_factor,
                 initial_cognitive_coefficient, final_cognitive_coefficient,
                 initial_social_coefficient, final_social_coefficient,
                 population_size, problem, boundaries):
        self.population = np.asarray([[]] * population_size)
        self.velocities = np.zeros((population_size, problem.num_dimensions))

        self.personal_bests = np.asarray([[]] * population_size)
        self.personal_bests_fitnesses = np.asarray([] * population_size)
        self.local_bests = np.asarray([[]] * population_size)
        self.local_bests_fitnesses = np.asarray([] * population_size)

        # Non-existent links should be set as -math.inf
        self.topology = np.ones((population_size, population_size))

        self.initial_inertia_factor = initial_inertia_factor
        self.final_inertia_factor = final_inertia_factor
        self.inertia_factor = self.initial_inertia_factor

        self.initial_cognitive_coefficient = initial_cognitive_coefficient
        self.final_cognitive_coefficient = final_cognitive_coefficient
        self.initial_social_coefficient = final_social_coefficient
        self.final_social_coefficient = final_social_coefficient

        self.cognitive_coefficient = initial_cognitive_coefficient
        self.social_coefficient = initial_social_coefficient

        self.boundaries = boundaries
        self.problem = problem

    def init_array(self, arr):
        return ((self.boundaries[1] - self.boundaries[0]) *
                np.random.random_sample((self.problem.num_dimensions,
                                         ))) + self.boundaries[0]

    def define_local_bests(self, arr):
        max_index = np.argmax(arr)
        if max_index < arr.shape[0] - 1:
            return np.append(self.personal_bests[max_index], arr[max_index])

    def update_local_bests(self):
        filtered_fitnesses = self.topology * self.personal_bests_fitnesses
        concatenated_fitnesses = np.concatenate((filtered_fitnesses,
                                                 self.local_bests_fitnesses.reshape(self.population.shape[0], 1)),  # noqa
                                                axis=1)
        new_local_bests = np.apply_along_axis(func1d=self.define_local_bests,
                                              axis=1,
                                              arr=concatenated_fitnesses)
        split = np.split(new_local_bests,
                         indices_or_sections=[0, self.problem.num_dimensions],
                         axis=1)
        self.local_bests = split[1]
        self.local_bests_fitnesses = split[2]

    def init_population(self):
        self.population = np.apply_along_axis(func1d=self.init_array,
                                              axis=1,
                                              arr=self.population)
        self.fitnesses = np.apply_along_axis(func1d=self.problem.evaluate_function,
                                             axis=1,
                                             arr=self.population)
        self.personal_bests = np.copy(self.population)
        self.personal_bests_fitnesses = np.copy(self.fitnesses)
        self.local_bests = np.copy(self.fitnesses)
        self.local_bests_fitnesses = np.copy(self.fitnesses)

    def update_velocity(self):
        self.velocities += self.inertia_factor * self.velocities
        self.velocities += self.cognitive_coefficient *\
            np.random.random_sample() *\
            (self.personal_bests - self.population)
        self.velocities += self.social_coefficient *\
            np.random.random_sample() *\
            (self.local_bests - self.population)

    def update_position(self):
        self.population += self.velocities

if __name__ == '__main__':
    problem = SphereFunction(2)
    pso = PSO(0, 0, 0, 0, 0, 0, 10, problem, (-10, 10))
    pso.init_population()
    pso.update_local_bests()
    pso.update_velocity()
    pso.update_position()
    print(pso.population)
