import numpy as np
import warnings
import random
from matplotlib import pyplot as plt
from fitnesses import *

class Organism: # Lineages > Organisms
    def __init__(self, name: str, phenotype: list, parent: object, pop: object) -> None:
        if not type(parent) == Organism or not type(pop) == Population:
            if parent is not None:
                raise Exception(f'parent must be of type Organism\npop must be of type Population\nparent is type {type(parent)}\npop is type {type(pop)}')
        self.name = name
        self.phenotype = phenotype
        self.parent = parent
        self.children: list = []
        self.mutations: list = []
        self.pop = pop

    def __str__(self) -> str:
        return self.name

    def get_fitness(self) -> float:
        return self.pop.sim.fitness_func(FitnessFunctions, self.phenotype)

    def reproduce(self, count: int) -> object:
        child = Organism(name=f'Gen {len(pop.organisms)}, Org {count}', phenotype=self.phenotype, parent=self, pop=self.pop)
        self.children.append(child)
        for mutation in child.mutations:
            mutation.victims.append(child)
        return child

    def mutate(self, mean: float, sd: float) -> None:
        impact = np.random.normal(mean, sd, len(self.phenotype))
        id = f'MO: {self.name}'
        mutation = Mutation(id, victims=[self], impact=impact)
        self.mutations.append(mutation)
        return


class Mutation:
    def __init__(self, id: str, victims: list, impact: list) -> None:
        self.id = id
        self.victims = victims
        self.impact = impact

    def __str__(self) -> str:
        return self.id

    def plot_fixation(self, *args, **kwargs):
        return


class Population: #TODO: write reproduction choosing mutations with numpy arrays, poisson random sample for pruning population to n
    def __init__(self, n: int, name: str, Ub: float, b_mean: float, b_sd: float, Ud: float, d_mean: float,
                 d_sd: float, len_genome: int) -> None:
        self.n = n
        self.name = name
        self.Ub = Ub
        self.b_mean = b_mean
        self.b_sd = b_sd
        self.Ud = Ud
        self.d_mean = d_mean
        self.d_sd = d_sd
        self.organisms = [[]]
        self.sim = None
        self.len_genome = len_genome
        for i in range(n):
            self.organisms[0].append(Organism(f'Gen 0, Org {i}', np.random.random(self.len_genome), None, self))
            pass

    def __str__(self) -> str:
        return self.name

    def reproduce(self) -> None:
        self.organisms.append([])
        random.shuffle(self.organisms[-2])
        # randomly distribute beneficial mutations
        num_beneficial_mutations = np.random.binomial(self.n, self.Ub)
        for i in range(num_beneficial_mutations):
            self.organisms[-2][i].mutate(self.b_mean, self.b_sd)
        # randomly distribute deleterious mutations
        random.shuffle(self.organisms[-2])
        num_deleterious_mutations = np.random.binomial(self.n, self.Ud)
        print(num_deleterious_mutations)
        for i in range(num_deleterious_mutations):
            self.organisms[-2][i].mutate(self.d_mean, self.d_sd)

        # reproduce according to fitness with key:value = org:num_offspring before sampling
        new_gen = {}
        for org in self.organisms[-2]:
            fitness = org.get_fitness()
            new_gen[org] = round(np.exp(fitness))  # f(t) = f(0) * exp(fitness * t)

        # randomly delete organisms until population is correct
        while sum(new_gen.values()) > self.n:
            new_gen[random.choices(list(new_gen.keys()))[0]] -= 1

        # populate next generation
        for org, children in list(new_gen.items()):
            count = 0
            for i in range(children):
                self.organisms[-1].append(org.reproduce(count))
                count += 1

    def plot(self, img_destination: str, *args, **kwargs) -> None:
        return


class Simulation:
    def __init__(self, pop: Population, runs: int, gens: int, fitness_func) -> None:
        self.pop = pop
        self.runs = runs
        self.gens = gens
        self.fitness_func = fitness_func
        self.compiled_fitnesses = np.zeros((runs, gens, self.pop.n))

    def __str__(self) -> str:
        return f'Instance of Simulation with {self.pop}, {self.runs} runs, and {self.gens} gens'

    def run(self):
        for i in range(self.runs):
            for j in range(self.gens):
                self.pop.reproduce()
            j = 0
            while j < self.gens:
                k = 0
                while k < self.pop.n:
                    self.compiled_fitnesses[i][j][k] = self.pop.organisms[j][k].get_fitness()
                    k += 1
                j += 1
            self.pop.organisms = [[]]

    def fitness(self, phenotype: list) -> float:
        return self.fitness_func(*phenotype)

    def plot(self, img_destination: str, *args, **kwargs) -> None:
        return


if __name__ == '__main__':
    runs = 1
    gens = 30

    pop = Population(10 ** 2, "pop_0", 10 ** (-3), 0.01, 0.002, 0.1, 0.02, 0.004, 3)
    sim = Simulation(pop, runs, gens, FitnessFunctions.nvariate_converge);
    pop.sim = sim

    print(pop)
    print(pop.sim)
    print(str(pop.organisms[0][0]), "through", str(pop.organisms[-1][-1]))
    print('hello world')
    sim.run()

    raw_fitnesses = sim.compiled_fitnesses
    averages = np.zeros((raw_fitnesses.shape[1:]))
    print(averages.shape, raw_fitnesses.shape)
    i = 0
    while i < raw_fitnesses.shape[1]:
        j = 0
        while j < raw_fitnesses.shape[2]:
            print(raw_fitnesses.shape[0], i, j)
            averages[i][j] = sum(raw_fitnesses[:, i, j]) / raw_fitnesses.shape[0]
            j += 1
        i += 1
    print(averages, averages.shape)
    plt.imshow(averages)
    plt.colorbar()
    plt.title("Average Fitness Across Generations")
    plt.savefig(f'imgs/Full: {runs} runs, {gens} gens')