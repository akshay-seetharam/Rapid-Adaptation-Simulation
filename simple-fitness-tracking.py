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

    start_time = time.time()

    population = 1000
    generations = 150
    starting_fitness = 0.5
    b = 0.01
    d =- -1 * 0.05

    Ubs = [round(i * 1000)/1000 for i in np.linspace(10**-3, 10**-2, 3)]
    Uds = [round(i*100)/100 for i in np.linspace(10**-2, 10**-1, 3)]

    plt.figure(figsize=(8, 6), dpi=80)

    fig, axs = plt.subplots(nrows=len(Ubs), ncols=len(Uds))
    # set the spacing between subplots
    plt.subplots_adjust(wspace=0.5,
                        hspace=0.5)
    plt.suptitle(f'Mean Fitness vs. Generation (b={b}, d={d})')
    for i, Ub in enumerate(Ubs):
        for j, Ud in enumerate(Uds):
            fitnesses = [{starting_fitness: population}]  # key: value, fitness: # of individuals with fitness

            for _ in range(generations):
                next_gen(population, fitnesses, Ub, b, Ud, d)
                print(f'gen {_} done')

            mean_fitnesses = []
            for generation in fitnesses:
                # print(generation)
                mean_fitnesses.append(sum([fitness * size / population for fitness, size in generation.items()]))

            axs[i][j].scatter(range(len(fitnesses)), mean_fitnesses)
            axs[i][j].set_title(f'Ub={Ub}, Ud={Ud}')
    plt.savefig(f'imgs/{generations} generations sfp')
    print("--- %s seconds ---" % (time.time() - start_time))