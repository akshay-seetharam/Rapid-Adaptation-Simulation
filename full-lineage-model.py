import numpy as np
import random
import matplotlib.pyplot as plt
from alive_progress import alive_bar
import math
import sys

class Mutation:
    def __init__(self, impact: float):
        self.impact = impact

    def __str__(self):
        return self.id

class Lineage:
    def __init__(self, mutations: set, fitness: float, Ub: float, b_mean: float, b_stdev: float, epistasis: float, func):
        self.mutations = mutations
        self.fitness = fitness
        self.Ub = Ub
        self.b_mean = b_mean
        self.b_stdev = b_stdev
        self.epistasis = epistasis
        self.func = func

    def mutate(self):
        mutation = Mutation(np.random.normal(self.b_mean, self.b_stdev))
        mutations = self.mutations.union({mutation})
        return Lineage(mutations, self.fitness + mutation.impact, self.Ub * self.epistasis, self.b_mean, self.b_stdev, self.epistasis, self.func) # epistasis by changing mutation parameters

class Population:
    def __init__(self, size: int, fitness: float, Ub:float, b_mean: float, b_stdev: float, epistasis: float, func):
        self.size = size
        self.starting_fitness = fitness
        self.Ub = Ub
        self.b_mean = b_mean
        self.b_stdev = b_stdev
        self.epistasis = epistasis
        self.generations = [{Lineage(set(), fitness, self.Ub, b_mean, b_stdev, epistasis, func): size * 0.999, Lineage(set(), fitness + 0.04, self.Ub, b_mean, b_stdev, epistasis, func): size * 0.001}] # Where starting lineages are found
        self.fitnesses = np.zeros((1, size))
        self.fitnesses[0] = [self.starting_fitness] * self.size

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
            population_growth = round(np.exp(lineage.func(lineage.fitness)) * count)
            self.generations[-1][lineage] = population_growth

        # mutate a proportion of each lineage
            num_mutated = round(np.random.binomial(np.exp(lineage.func(lineage.fitness)) * count, self.Ub))
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

def stabilizer(x):
    return 1 - (x - 0.75) ** 2

def bimodal(x):
    return 1 - (x - 0.6) ** 2 + 1 - (x - 1.2) ** 2

def directional(x):
    return x

if __name__ == '__main__':
    """ PARAMETERS """
    runs = 30
    num_gens = 200
    size = 10**4
    starting_fitness = 1
    Ub = 0
    b_mean = 0.1
    b_stdev = 0.05
    epistasis = 1
    func = directional
    """ PARAMETERS """

    sim = Simulation(runs, num_gens, size, starting_fitness, Ub, b_mean, b_stdev, epistasis, func)
    with alive_bar(runs * num_gens, bar='notes') as bar:
        for _ in sim.run():
            bar()

    print("Simulation Complete")
    # style = random.choice(plt.style.available)
    style = 'dark_background'
    plt.style.use(style)

    for i in range(runs):
        if sim.compiled_fitnesses[i][num_gens - 1][int(size * 0.5)] > starting_fitness:
            print(f"Run {i} Fixed")
        else:
            print(f"Run {i} Not Fixed")
        plt.imshow(sim.compiled_fitnesses[i], aspect='auto', interpolation='none')
        plt.title(f"{num_gens} Gens, {size} Size, {starting_fitness} Starting Fitness, {Ub} Ub, {b_mean} b_mean, {b_stdev} b_stdev, {epistasis} epistasis\nStyle: {style}")
        plt.xlabel("Organismal Index")
        plt.ylabel("Generation")
        plt.colorbar()
        # plt.show()
        plt.savefig(f'imgs/plot_{i}.png')
    sys.exit(0)