import math
import numpy as np
from functions import SphereFunction


class PSO:

    def __init__(self, initial_inertia_factor, final_inertia_factor,
                 initial_cognitive_coefficient, final_cognitive_coefficient,
                 initial_social_coefficient, final_social_coefficient,
                 population_size, problem, boundaries, max_iterations):
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
        self.initial_social_coefficient = initial_social_coefficient
        self.final_social_coefficient = final_social_coefficient

        self.cognitive_coefficient = initial_cognitive_coefficient
        self.social_coefficient = initial_social_coefficient

        self.boundaries = boundaries
        self.problem = problem

        self.max_iterations = max_iterations
        self.current_iteration = 0

    def init_algorithm(self):
        self.init_population()
        self.update_local_bests()

    def run(self):
        for current_iteration in range(0, self.max_iterations):
            print(current_iteration)
            self.update_velocity()
            self.update_position()
            self.update_personal_bests()
            self.update_local_bests()
            self.update_coefficients()
        print(np.max(self.local_bests_fitnesses))

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

    def define_personal_bests(self, arr):
        current_position = arr[:self.problem.num_dimensions]
        current_fitness = arr[self.problem.num_dimensions]
        personal_best_position = arr[self.problem.num_dimensions + 1:(2 * self.problem.num_dimensions) + 1]  # noqa
        personal_best_fitness = arr[(2 * self.problem.num_dimensions) + 1:(2 * self.problem.num_dimensions) + 2]  # noqa
        if current_fitness > personal_best_fitness:
            return np.append(current_position, current_fitness)
        else:
            return np.append(personal_best_position, personal_best_fitness)

    def update_personal_bests(self):
        current_fitnesses = np.apply_along_axis(func1d=self.problem.evaluate_function,  # noqa
                                                axis=1,
                                                arr=self.population)
        data = np.concatenate((self.population, current_fitnesses.reshape(self.population.shape[0], 1)), axis=1)  # noqa
        data = np.concatenate((data, self.personal_bests), axis=1)
        data = np.concatenate((data, self.personal_bests_fitnesses.reshape(self.population.shape[0], 1)), axis=1)  # noqa
        new_personal_bests = np.apply_along_axis(func1d=self.define_personal_bests,  # noqa
                                                 axis=1,
                                                 arr=data)
        split = np.split(new_personal_bests,
                         indices_or_sections=[0, self.problem.num_dimensions],
                         axis=1)
        self.personal_bests = split[1]
        self.personal_bests_fitnesses = split[2].reshape(self.population.shape[0])

    def init_population(self):
        self.population = np.apply_along_axis(func1d=self.init_array,
                                              axis=1,
                                              arr=self.population)
        self.fitnesses = np.apply_along_axis(func1d=self.problem.evaluate_function,  # noqa
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

    def limit_individual_position(self, arr):
        if arr[0] < self.boundaries[0]:
            return [self.boundaries[0], -1 * arr[1]]
        elif arr[0] > self.boundaries[1]:
            return [self.boundaries[1], -1 * arr[1]]
        else:
            return arr

    def limit_individual_positions(self, arr):
        positions = arr[:self.problem.num_dimensions]
        velocities = arr[self.problem.num_dimensions:]
        pos_vel_array = np.transpose([positions, velocities])
        new_pos_vel_array = np.apply_along_axis(func1d=self.limit_individual_position,
                                                axis=1,
                                                arr=pos_vel_array)
        pos_vel_array = np.transpose(new_pos_vel_array)
        return np.concatenate((pos_vel_array[0], pos_vel_array[1]))

    def update_position(self):
        self.population += self.velocities
        concatenated_pos_and_vel = np.concatenate((self.population,
                                                   self.velocities), axis=1)
        new_population_and_vel = np.apply_along_axis(func1d=self.limit_individual_positions,
                                                     axis=1,
                                                     arr=concatenated_pos_and_vel)
        split = np.split(new_population_and_vel,
                         indices_or_sections=[0, self.problem.num_dimensions],
                         axis=1)
        self.population = split[1]
        self.velocities = split[2]

    def update_coefficients(self):
        self.inertia_factor += float(self.final_inertia_factor -
                                     self.initial_inertia_factor) /\
            self.max_iterations
        self.cognitive_coefficient += float(self.final_cognitive_coefficient -
                                            self.initial_cognitive_coefficient) /\
            self.max_iterations
        self.social_coefficient += float(self.final_social_coefficient -
                                         self.initial_social_coefficient) /\
            self.max_iterations

if __name__ == '__main__':
    problem = SphereFunction(2)
    pso = PSO(0.9, 0.4, 4, 0, 0, 4, 10, problem, (-10, 10), 10)
    pso.init_algorithm()
    pso.run()
