import random
from math import exp, sqrt
import time
from matplotlib import pyplot as plt
import numpy as np
import alive_progress
from alive_progress.styles import showtime, Show

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
        # sampled[fitness + b] = Ub * reproduced[fitness] # TODO: one-by-one sampling is exactly the same as binomial
        beneficial_mutations = np.random.binomial(reproduced[fitness], Ub)
        detrimental_mutations = np.random.binomial(reproduced[fitness], Ud)
        try:
            sampled[fitness + b] += beneficial_mutations
        except KeyError as ke:
            # print(ke)
            sampled[fitness + b] = beneficial_mutations
        try:
            sampled[fitness + d] += detrimental_mutations
        except KeyError as ke:
            # print(ke)
            sampled[fitness + d] = detrimental_mutations
        sampled[fitness] = max(0, reproduced[fitness] - (beneficial_mutations + detrimental_mutations))
        # pprint(sampled)

    # bring population back down to simulate a sample from one flask being drawn into another maintaining constant population
    try:
        correction_factor = 1.0 * population / sum(sampled.values())
    except Exception as e:
        # print(e)
        # print(sampled)
        return
    for fitness in sampled:
        sampled[fitness] *= correction_factor

    fitnesses.append(sampled)

def simulate(population, generations, starting_fitness, b, d, Ubs, Uds, runs, plot=False):
    if plot:
        plt.figure(figsize=(1000, 1000), dpi=80)
        fig, axs = plt.subplots(nrows=len(Ubs), ncols=len(Uds))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.suptitle(f'Mean Fitness vs. Generation (pop.={population}, b={b}, d=–{-1 * d})')

    for i, Ub in enumerate(Ubs):
        for j, Ud in enumerate(Uds):
            avgs_of_avgs = []
            for run in range(runs):
                fitnesses = [{starting_fitness: population}]
                for _ in range(generations):
                    next_gen(population, fitnesses, Ub, b, Ud, d)
                    yield
                mean_fitnesses = []
                for generation in fitnesses:
                    mean_fitnesses.append(sum([fitness * size / population for fitness, size in generation.items()]))
                if plot:
                    differences = [0]
                    k = 1
                    while k < len(mean_fitnesses):
                        differences.append(mean_fitnesses[k] - mean_fitnesses[k - 1])
                        k += 1

                    axs[i][j].plot(range(len(fitnesses)), differences, color='gray')
                    axs[i][j].set_title(f'Ub={Ub}, Ud={Ud}')

            x = np.linspace(0, generations, 1000)
            y1 = [population * Ub * b**2 * i for i in x]
            y2 = [(b**2 * ((2 * np.log(population * b) - np.log(b / Ub)) / (np.log(b / Ub))**2)) * i for i in x]
            axs[i][j].plot(x, y2, label='Common Beneficial Mutations', color='red')
            axs[i][j].plot(x, y1, label='Successive Mutations', color='blue')
            axs[i][j].legend()



            print(f'Run #{run} done for Ub = {Ub} and Ud = –{-1 * Ud}')
    if plot:
        plt.savefig(f'imgs/{generations} generations, {runs} runs')

if __name__=='__main__':

    start_time = time.time()

    population = 100
    generations = 30
    starting_fitness = 0.5
    b = 0.01
    d = -0.0 # make d = 0 when comparing against desai/fisher eqns for mean fitness growth
    Ubs = [round(i * 1000)/1000 for i in np.linspace(10**-2, 10**-1, 2)]
    Uds = [round(i*100)/100 for i in np.linspace(10**-2, 10**-1, 2)]

    runs = 15
    with alive_progress.alive_bar(runs * len(Ubs) * len(Uds) * generations, bar='notes') as bar:
        for i in simulate(population, generations, starting_fitness, b, d, Ubs, Uds, runs, plot=True):
            # print(i)
            bar()
    print("--- %s seconds ---" % (time.time() - start_time))