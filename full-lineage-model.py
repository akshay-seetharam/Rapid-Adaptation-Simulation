import numpy as np
import warnings
import random
from matplotlib import pyplot as plt
from fitnesses import *

class Mutation:
    def __init__(self, id: str, victims: list, impact: list) -> None:
        self.id = id
        self.victims = victims
        self.impact = impact

    def __str__(self) -> str:
        return self.id

    def plot_fixation(self, *args, **kwargs) -> None:
        return

class Lineage:
    def __init__(self, mutations: set, population: Population) -> None:
        self.mutations = mutations
        self.population = population
        self.organisms = np.zeros((population.simulation.gens, population.n, num_traits)) # each generation has organisms represented by a phenotype with a num_traits-dimensional phenotype; the fitness function can be used to calculate the fitness

    def __str__(self) -> str:
        return f'Lineage with mutations: {self.mutations}'

    def get_fitness(self) -> np.ndarray:
        fitness = np.zeros_like(self.organisms)
        i = 0
        while i < self.organisms.shape[0]
            j = 0
            while j < self.organisms.shape[1]
                fitness[i, j] = self.population.simulation.fitness_func(FitnessFunctions, self.organisms[i, j])
            i += 1
        return fitness

    def get_org_fitness(self, org_id: list) -> float:
        return self.population.simulation.fitness_func(FitnessFunctions, self.organisms[org_id])

class Population:
    def __init__(self, n: int, name: str, Ub: float, b_mean: float, b_sd: float, Ud: float, d_mean: float,
                 d_sd: float, num_traits: int, sim: Simulation) -> None:
        self.n = n
        self.name = name
        self.Ub = Ub
        self.b_mean = b_mean
        self.b_sd = b_sd
        self.Ud = Ud
        self.d_mean = d_mean
        self.d_sd = d_sd
        self.num_traits = num_traits
        self.sim = sim

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