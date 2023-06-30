import numpy as np
from matplotlib import pyplot as plt

def average_fitness(genome):
    return np.average(np.sum(genome, axis=1))

def horizontal_gene_transfer(genome):
    #TODO Implementation (ask for bio details)
    return genome

def reproduce(genome):
    #TODO Reproduce based on fitness
    new_genome = np.zeros_like(genome)
    weights = np.exp(np.sum(genome, axis=1))
    weights /= np.sum(weights)
    for i in range(new_genome.shape[0]):
        new_genome[i] = genome[np.random.choice(genome.shape[0], p=weights)]
        # weighted average with exponential of fitnesses
    #TODO Mutate some
    global mutation_prob
    global population
    num_mutants = int(population * mutation_prob)
    for i in range(num_mutants):
        print('Mutating')
        new_polymorphic_site = np.zeros((population, 1))
        new_polymorphic_site[np.random.choice(population)][0] = 1
        new_genome = np.concatenate((new_genome, new_polymorphic_site), axis=1) # debug this line
    return new_genome

def reproductive_update(genome):
    genome = horizontal_gene_transfer(genome)
    genome = reproduce(genome)
    return genome

if __name__ == '__main__':
    ### PARAMS ##
    population = 1000
    polymorphic_sites = 5
    U_b = 0.1
    activated_proportion = 0.05
    generations = 20
    mutation_prob = 10 ** -3
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

    print(genome)
    

