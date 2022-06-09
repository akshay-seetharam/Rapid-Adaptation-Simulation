import random
from math import exp
import time

def next_gen(population, fitnesses, Ub, b, Ud, d):
    reproduced = {}
    for fitness in fitnesses[-1]:
        # have the bacteria grow by f(t) = e^xt
        reproduced[fitness] = fitnesses[-1][fitness] * exp(fitness)
    # mutate some of the bacteria positively and some negatively
    sampled = {}
    for fitness in reproduced:
        sampled[fitness + b] = Ub * reproduced[fitness]
        sampled[fitness + d] = Ud * reproduced[fitness]
        sampled[fitness] = reproduced[fitness] - (sampled[fitness + b] + sampled[fitness + d])

    # bring population back down to simulate a sample from one flask being drawn into another maintaining constant population
    correction_factor = population / sum(sampled.values())
    for fitness in sampled:
        sampled[fitness] *= correction_factor

    fitnesses.append(sampled)


if __name__=='__main__':
    # This uses the format of only keeping track of fitness quantities, disregarding the organism and genome level
    population = 10 ** 9
    generations = 10
    starting_fitness = 0.5
    Ub = 0.01 # probability of beneficial mutation
    b = 0.01 # affect of beneficial mutation to fitness
    Ud = 0.1 # probability of detrimental mutation
    d = -1 * 0.02 # affect of detrimental mutation to fitness


    fitnesses = [{starting_fitness: population}]

    start_time = time.time()

    for i in range(generations):
        next_gen(population, fitnesses, Ub, b, Ud, d)

    print(fitnesses)

    print("--- %s seconds ---" % (time.time() - start_time))