import numpy as np
import random

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

    def reproduce(self) -> None:
        self.generations.append({})

        # reproduce each lineage in the previous generation
        for lineage, count in self.generations[-2].items():
            population_growth = np.exp(lineage.fitness) * count
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

        print('reproduction done for generation', len(self.generations) - 1)
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
                # self.compiled_fitnesses[run] = population.fitnesses # TODO: make the population.fitnesses return what it is supposed to

if __name__ == '__main__':
    sim = Simulation(15, 100, 10**4, 0.5, 10**(-3), 0.01, 0.005)
    sim.run()
    print("Simulation Complete")