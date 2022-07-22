import numpy as np
import random
import matplotlib.pyplot as plt
from alive_progress import alive_bar

class Mutation:
    def __init__(self, impact: float):
        self.impact = impact

    def __str__(self):
        return self.id

class Lineage:
    def __init__(self, mutations: set, fitness: float, Ub: float, b_mean: float, b_stdev: float):
        self.mutations = mutations
        self.fitness = fitness
        self.Ub = Ub
        self.b_mean = b_mean
        self.b_stdev = b_stdev

    def mutate(self):
        mutation = Mutation(np.random.normal(self.b_mean, self.b_stdev))
        mutations = self.mutations.union({mutation})
        return Lineage(mutations, self.fitness + mutation.impact, self.Ub, self.b_mean, self.b_stdev)

class Population:
    def __init__(self, size: int, fitness: float, Ub:float, b_mean: float, b_stdev: float):
        self.size = size
        self.starting_fitness = fitness
        self.Ub = Ub
        self.b_mean = b_mean
        self.b_stdev = b_stdev
        self.generations = [{Lineage(set(), fitness, Ub, b_mean, b_stdev): size}]
        self.fitnesses = np.zeros((1, size))
        j = 0
        while j < self.size:
            self.fitnesses[0, j] = self.starting_fitness
            j += 1

    def update_fitnesses(self) -> None:
        if len(self.generations) != self.fitnesses.shape[0] + 1:
            raise Exception('self.generations and self.fitnesses do not have compatibile shapes')
        self.fitnesses = np.append(self.fitnesses, np.zeros((1, self.size)), axis=0)
        index = 0
        for lineage, count in self.generations[-1].items():
            for _ in range(count):
                self.fitnesses[-1, index] = lineage.fitness
                index += 1


    def reproduce(self) -> None:
        self.generations.append({})

        # reproduce each lineage in the previous generation
        for lineage, count in self.generations[-2].items():
            population_growth = int(np.exp(lineage.fitness) * count)
            self.generations[-1][lineage] = population_growth

        # mutate a proportion of each lineage
            num_mutated = int(np.random.binomial(count * np.exp(lineage.fitness), self.Ub))
            for i in range(num_mutated):
                new_lineage = lineage.mutate()
                self.generations[-1][new_lineage] = 1
                self.generations[-1][lineage] -= 1


        # remove the lineages with 0 population to prevent any chicanery
        self.generations[-1] = {k: v for k, v in self.generations[-1].items() if v > 0}

        # prune population down to size
        while sum(self.generations[-1].values()) > self.size:
            choice = random.choices(list(self.generations[-1].keys()), weights=list(self.generations[-1].values()))[0]
            self.generations[-1][choice] -= 1

        # remove the lineages with 0 population to prevent any chicanery
        self.generations[-1] = {k: v for k, v in self.generations[-1].items() if v > 0}

        self.update_fitnesses()

class Simulation:
    def __init__(self, runs: int, num_gens: int, size: int, *args):
        self.runs = runs
        self.num_gens = num_gens
        self.size = size
        self.compiled_fitnesses = np.zeros((runs, num_gens + 1, size))
        self.population_parameters = [self.size, *args]

    def run(self) -> None:
        for run in range(self.runs):
            population = Population(*self.population_parameters)
            for _ in range(self.num_gens):
                population.reproduce()
                yield
            self.compiled_fitnesses[run] = population.fitnesses

if __name__ == '__main__':
    """ CONSTANTS """
    runs = 1
    num_gens = 300
    size = 10**4
    starting_fitness = 0.5
    Ub = 10 ** -3
    b_mean = 0.1
    b_stdev = 0.005
    """ CONSTANTS """
    sim = Simulation(runs, num_gens, size, starting_fitness, Ub, b_mean, b_stdev)
    with alive_bar(runs * num_gens) as bar:
        for _ in sim.run():
            bar()

    print("Simulation Complete")
    plt.imshow(sim.compiled_fitnesses[0], aspect='auto')
    plt.colorbar()
    plt.show()