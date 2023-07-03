import numpy as np
from matplotlib import pyplot as plt

def average_fitness(genome):
    return np.average(np.sum(genome, axis=1)) #TODO ask about what an appropriate function here is, because right now there's no consistent fitness improvement

def horizontal_gene_transfer(genome):
    #TODO Implementation (ask for bio details)
    # source retains allele, recipient receives allele from source
    return genome

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
    population = 1000
    polymorphic_sites = 5
    U_b = 0.1
    activated_proportion = 0.05
    generations = 50
    mutation_prob = 10 ** -3
    ### PARAMS ###

    genome = np.zeros((population, polymorphic_sites))

    for i in range(population):
        genome[i] = np.array([np.random.binomial(1, activated_proportion) for _ in range(polymorphic_sites)])

    old_fitness = average_fitness(genome)
    fitness_deltas = []

    for gen in range(generations):
        genome = reproductive_update(genome)
        new_fitness = average_fitness(genome)
        fitness_deltas.append(new_fitness - old_fitness)
        old_fitness = new_fitness

    plt.plot(range(generations), fitness_deltas)
    plt.xlabel('Generation')
    plt.ylabel('Delta Fitness')
    plt.title('Change in Fitness by Generation')
    plt.show()

    print(genome, genome.shape)
    

