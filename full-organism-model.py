import numpy as np
import warnings

class FitnessFunctions:
    def basic(cls, fitness: float) -> float:
        return fitness


class Organism:
    def __init__(self, name: str, genotype: list) -> None:
        self.name = name
        self.genotype = genotype

    def __str__(self) -> str:
        return self.name

    def get_fitness(self) -> float:
        return 0.1

    def mutate(self) -> None:
        return


class Mutation:
    def __init__(self, id: str, victims: list) -> None:
        self.id = id
        self.victims = victims

    def __str__(self) -> str:
        return self.id

    def plot_fixation(self, *args, **kwargs):
        return


class Population:
    def __init__(self, n: int, name: str, Ub: float, b_mean: float, b_sd: float, Ud: float, d_mean: float, d_sd: float) -> None:
        self.n = n
        self.name = name
        self.Ub = Ub
        self.b_mean = b_mean
        self.b_sd = b_sd
        self.Ud = Ud
        self.d_mean = d_mean
        self.d_sd = d_sd
        self.organisms = list
        for i in range(n):
            # self.organisms.append(Organism())
            # raise Exception("write this, idiot")
            pass

    def __str__(self) -> str:
        return self.name

    def reproduce(self) -> None:
        return

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
        return 0.1

    def plot(self, img_destination: str, *args, **kwargs) -> None:
        return


if __name__ == '__main__':
    pop = Population(10 ** 1, "pop_0", 10 ** (-3), 0.01, 0.002, 0.1, 0.02, 0.004)
    sim = Simulation(pop, 15, 50, FitnessFunctions.basic)
    org = Organism("org_0", [0.1, 0.2, 0.3])

    print(pop, sim, org)
    print('hello world')
