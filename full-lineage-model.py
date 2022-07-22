import numpy as np

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

    def mutate(self) -> Lineage:
        mutation = Mutation(np.random.normal(b_mean, b_stdev))
        mutations = self.mutations.union(mutation)
        return Lineage(mutations, self.fitness + mutation.impact, self.Ub, self.b_mean, self.b_stdev)

class Population:
    def __init__(self, size: int, fitness: float, Ub:float, b_mean: float, b_stdev: float):
        self.size = size
        self.starting_fitness = fitness
        self.Ub = Ub
        self.b_mean = b_mean
        self.b_stdev = b_stdev
        self.generations = [{Lineage(set(), fitness, Ub, b_mean, b_stdev): size}]

    def reproduce(self) -> None:
        pass

class Simulation:
    def __init__(self, runs: int, num_gens: int, size: int, *args):
        self.runs = runs
        self.num_gens = num_gens
        self.size = size
        self.compiled_fitnesses = np.zeros((runs, num_gens, size))
        self.population_parameters = [self.size, *args]

    def run(self) -> None:
        for run in range(self.runs):
            population = Population(*self.population_parameters)
            for gen in range(self.num_gens):
                population.reproduce()
                self.compiled_fitnesses[run] = population.fitnesses # TODO: make the population.fitnesses return what it is supposed to