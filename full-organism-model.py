import numpy as np
import warnings
import random


class FitnessFunctions:
    def univariate_basic(cls, fitness: float) -> float:
        """returns input"""
        return fitness

    def univariate_converge(self, fitness: float) -> float:
        """fitness function where 0.1 is the optimal fitness"""
        return 1 - (fitness - 0.1) ** 2


class Organism:
    def __init__(self, name: str, genotype: list, parent: Organism) -> None:
        self.name = name
        self.genotype = genotype
        self.parent = parent
        self.children = []
        self.mutations = []

    def __str__(self) -> str:
        return self.name

    def get_fitness(self) -> float:
        return 0.1

    def reproduce(self) -> Organism:
        child = Organism(name=None, genotype=self.genotype, parent=self)
        self.children.append(child)
        return child

    def mutate(self, mean: float, sd: float) -> None:
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


class Population:
    def __init__(self, n: int, name: str, Ub: float, b_mean: float, b_sd: float, Ud: float, d_mean: float,
                 d_sd: float) -> None:
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
        for i in range(n):
            # self.organisms[0].append(Organism())
            # raise Exception("write this, idiot")
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
        for i in range(num_deleterious_mutations):
            self.organisms[-2][i].mutate(self.d_mean, self.d_sd)

        # reproduce according to fitness with key:value = org:num_offspring before sampling
        new_gen = {}
        for org in self.organisms[-2]:
            fitness = org.get_fitness()
            new_gen[org] = np.exp(fitness)  # f(t) = f(0) * exp(fitness * t)

        # randomly delete organisms until population is correct
        while sum(new_gen.values()) > self.n:
            new_gen[random.choices(list(a.keys()))] -= 1

        # populate next generation
        for org, children in list(new_gen.items()):
            for i in range(children):
                self.organisms[-1].append(org.reproduce())

    def plot(self, img_destination: str, *args, **kwargs) -> None:
        return


class Simulation:
    def __init__(self, pop: Population, runs: int, gens: int, fitness_func) -> None:
        self.pop = pop
        self.runs = runs
        self.gens = gens
        self.fitness_func = fitness_func
        self.compiled_fitnesses = np.zeros((runs, gens, pop.n))

    def __str__(self) -> str:
        return f'Instance of Simulation with {self.pop}, {self.runs} runs, and {self.gens} gens'

    def fitness(self, genotype: list) -> float:
        return self.fitness_func(*genotype)

    def plot(self, img_destination: str, *args, **kwargs) -> None:
        return


if __name__ == '__main__':
    pop = Population(10 ** 1, "pop_0", 10 ** (-3), 0.01, 0.002, 0.1, 0.02, 0.004)
    sim = Simulation(pop, 15, 50, FitnessFunctions.univariate_basic);
    pop.sim = sim
    org = Organism("org_0", [0.1, 0.2, 0.3])

    print(pop, sim, org)
    print('hello world')
