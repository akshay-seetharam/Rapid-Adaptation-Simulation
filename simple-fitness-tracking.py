import random
from math import exp, sqrt
import time
from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint

def next_gen(population, fitnesses, Ub, b, Ud, d):
    reproduced = {}
    for fitness in fitnesses[-1]:
        # have the bacteria grow by f(t) = e^xt
        reproduced[fitness] = fitnesses[-1][fitness] * exp(fitness)
    # mutate some of the bacteria positively and some negatively
    # treating each bacteria's mutation as a bernoulli random event leads to the population's
    # mutations as a binomial random variable
    sampled = {}
    for fitness in reproduced:
        sampled[fitness + b] = Ub * reproduced[fitness] # TODO: one-by-one sampling is exactly the same as binomial
        # sampled[fitness + b] = round(np.random.normal(reproduced[fitness] * Ub, sqrt(reproduced[fitness]*Ub*(1-Ub))))
        sampled[fitness + d] = Ud * reproduced[fitness]
        # sampled[fitness + d] = round(np.random.normal(reproduced[fitness] * Ud, sqrt(reproduced[fitness]*Ud*(1-Ud))))
        sampled[fitness] = reproduced[fitness] - (sampled[fitness + b] + sampled[fitness + d])
        # pprint(sampled)

    # bring population back down to simulate a sample from one flask being drawn into another maintaining constant population
    try:
        correction_factor = 1.0 * population / sum(sampled.values())
    except Exception as e:
        print(e)
        print(sampled)
        return
    for fitness in sampled:
        sampled[fitness] *= correction_factor

    fitnesses.append(sampled)


if __name__=='__main__':
    # This uses the format of only keeping track of fitness quantities, disregarding the organism and genome level
    population = 100
    generations = 150
    starting_fitness = 0.5
    Ub = 0.01 # probability of beneficial mutation
    b = 0.01 # affect of beneficial mutation to fitness
    Ud = 0.02 # probability of detrimental mutation
    d = -1 * 0.05 # affect of detrimental mutation to fitness


    fitnesses = [{starting_fitness: population}] # key: value, fitness: # of individuals with fitness

    start_time = time.time()

    for i in range(generations):
        next_gen(population, fitnesses, Ub, b, Ud, d)
        print(f'gen {i} done')

    mean_fitnesses = []
    for generation in fitnesses:
        # print(generation)
        mean_fitnesses.append(sum([fitness * proportion for fitness, proportion in generation.items()]))

    print("--- %s seconds ---" % (time.time() - start_time))

    plt.scatter(range(len(fitnesses)), mean_fitnesses)
    plt.suptitle('Mean Fitness vs. Generation')
    plt.title(f'Ub={Ub}, b={b}, Ud={Ud}, d={d}')
    plt.ylabel('Population Mean Fitness')
    plt.xlabel('Generation')
    plt.show()