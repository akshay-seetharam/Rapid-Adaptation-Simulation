import numpy as np
from matplotlib import pyplot as plt

def average_fitness(genome):
    return np.average(np.sum(genome, axis = 1))

def horizontal_gene_transfer(genome):
    #TODO Implementation (ask for bio details)
    return genome

def reproduce(genome):
    #TODO Reproduce based on fitness
    #TODO Mutate some
    return genome

def prune_to_population(genome):
    #TODO Prune to
    return genome

def reproductive_update(genome):
    genome = horizontal_gene_transfer(genome)
    genome = reproduce(genome)
    genome = prune_to_population(genome, N)
    return genome

if __name__ == '__main__':
    ### PARAMS ##
    population = 1000
    polymorphic_sites = 5
    U_b = 0.1
    activated_proportion = 0.05
    generations = 20
    mutation_prob = 10 ** -4
    ### PARAMS ###

    genome = np.zeros((population, polymorphic_sites))

    for i in range(population):
        genome[i] = np.array([np.random.binomial(1, activated_proportion) for _ in range(polymorphic_sites)])

    fitnesses = [average_fitness(genome)]

    for gen in range(generations):
        genome = reproductive_update(genome)
        fitnesses.append(average_fitness(genome))

    plt.plot(range(generations + 1), fitnesses)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()
    

