import numpy as np

if __name__=='__main__':
    """ PARAMETERS """
    # taken from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4699269/table/T but with population cloud rather than single point
    n = 2 # dimension of phenotypic space
    a = 1.0 # robustness parameter in fitness function
    Q = 1.0 # epistasis parameter in fitness function
    s0 = 0.5 # maladaptaion of ancestor
    r = 0.05 # length of mutational vector
    sigma = 0.01 # standard deviation of normal deviates of the mutation vector on each axis
    def x(): # scaled distance to the optimum
        return r * np.sqrt(n) / (2 * d)
    e = 1.0 # epistasis defined as big long expression from table, click URL in comment above
    m = 1 # pleiotropy, number of phenotypes affected by each mutation
    N = 100 # number of individuals in population
    mu = 10 ** -3 # mutation rate

    def fitness(d): # fitness function from paper (input is distance from fitness peak)
        return np.exp(-1 * a * (d ** Q))

    generations = 100 # number of generations to run simulation for
    """ PARAMETERS """

    """ SIMULATION """
    # initialize population
    population = np.random.normal(1.5, 0.5, size=(N, n)) # random population
    fitnesses = np.zeros((generations, N))
    j = 0
    while j < fitnesses.shape[1]:
        fitnesses[0][j] = fitness(np.linalg.norm(population[j])) # calculate fitness of each individual
        j += 1
    print(population)
    print(fitnesses)