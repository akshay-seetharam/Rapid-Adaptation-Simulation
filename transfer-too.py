import numpy as np

if __name__ == '__main__':
    ### PARAMS ##
    population = 1000
    polymorphic_sites = 5
    U_b = 0.1
    activated_proportion = 0.05
    generations = 20

    ### END PARAMS ###

    genome = np.zeros((population, polymorphic_sites))

    for i in range(population):
        genome[i] = np.array(np.random.binomial(1, activated_proportion) for _ in range(polymorphic_sites))

    fitnesses = [average_fitness(genome)]

    for gen in range(generations):
        genome = reproductive_update(genome)
        fitnesses.append(average_fitness(genome))
    
    

