import numpy as np
from matplotlib import pyplot as plt
import random
import sys

def average_fitness(genome):
    return np.average(np.sum(genome, axis=1) * U_b)

def horizontal_gene_transfer(genome):
    # source retains allele, recipient receives allele from source
    new_genome = genome.copy()
    for _ in range(num_recombinations):
        donor = random.choice((0, genome.shape[0] - 1))
        locus = random.choice((0, genome.shape[1] - 1))
        recipient = donor
        while recipient == donor:
            recipient = random.choice((0, genome.shape[0] - 1))
        new_genome[recipient][locus] = new_genome[donor][locus]

    return new_genome

def reproduce(genome):
    new_genome = np.zeros_like(genome)
    weights = np.exp(np.sum(genome, axis=1))
    weights /= np.sum(weights)
    for i in range(new_genome.shape[0]):
        new_genome[i] = genome[np.random.choice(genome.shape[0], p=weights)]
        # weighted average with exponential of fitnesses
    global mutation_prob
    global population
    num_mutants = int(population * mutation_prob)
    for i in range(num_mutants):
        print('Mutating')
        new_polymorphic_site = np.zeros((population, 1))
        new_polymorphic_site[np.random.choice(population)][0] = 1
        new_genome = np.concatenate((new_genome, new_polymorphic_site), axis=1) # debug this line

    new_genome = np.delete(new_genome, np.argwhere(np.all(new_genome[..., :] == 0, axis=0)), axis=1)
    # stop tracking loci that are no longer polymorphic

    return new_genome

def reproductive_update(genome):
    genome = horizontal_gene_transfer(genome)
    genome = reproduce(genome)
    return genome

if __name__ == '__main__':
    ### PARAMS ##
    population = 10**4
    polymorphic_sites = 1
    U_b = 0.01
    activated_proportion = 0.01
    generations = 100
    mutation_prob = 10 ** -3
    num_recombinations = 100 # on order of beneficial mutation rate
    ### PARAMS ###

    genome = np.zeros((population, polymorphic_sites))

    for i in range(population):
        # genome[i] = np.array([np.random.binomial(1, activated_proportion) for _ in range(polymorphic_sites)])
        genome[i] = 0
    genome[0][0] = 1

    old_fitness = average_fitness(genome)
    fitness_deltas = []
    fitness_progression = []

    for gen in range(generations):
        genome = reproductive_update(genome)
        new_fitness = average_fitness(genome)
        fitness_progression.append(new_fitness)
        fitness_deltas.append(new_fitness - old_fitness)
        old_fitness = new_fitness
        print(f'Done with generation {gen}', sum(genome[:, 0]/population))
        if sum(genome[:, 0]) == population:
            print('Fixed')
            sys.exit()
        elif sum(genome[:, 0]) == 0:
            print('Extinct')
            sys.exit()

    plt.figure(figsize=(40, 40))

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].plot(range(generations), fitness_deltas, color='red')
    ax[0].set_xlabel('Generation')
    ax[0].set_ylabel('Delta Fitness')
    ax[0].set_title('Change in Fitness by Generation')

    ax[1].plot(range(generations), fitness_progression, color='red')
    ax[1].set_xlabel('Generation')
    ax[1].set_ylabel('Fitness')
    ax[1].set_title('Fitness by Generation')

    plt.savefig(f'{num_recombinations} recombinations.png')

    print(genome, genome.shape)

