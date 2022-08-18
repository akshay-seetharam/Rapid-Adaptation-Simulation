import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def select_parents():
    """
    Select the parents that will reproduce
    """
    # Calculate the fitnesses of the population
    chances = np.log([fitness(i) for i in population]) # get logarithmic fitnesses
    chances /= sum(chances) # normalize the chances
    # Select parents with replacement, weighted by fitness
    parents = []
    for i in range(N):
        choice= np.random.choice(range(N), p=chances) # TODO: Check with jimmy if I should use raw fitnesses or logarithmic fitnesses or something else
        parents.append(population[choice])

    return parents

def next_gen():
    """
    Calculate the next generation of the population
    """
    # select parents that will reproduce
    parents = select_parents()
    # mutate offspring
    for i in range(len(population)):
        vector = np.random.random((n)) # generate a random mutation vector
        vector *= r / np.linalg.norm(vector) # scale to r
        # population[i] = [parents[j] + vector[j] for j in range(n)] # mutate the individual
        population += vector

def plot_fitnesses():
    """
    Plot the fitnesses of the population
    """
    # Set the figure size
    plt.rcParams["figure.figsize"] = [12.0, 3.5]
    plt.rcParams["figure.autolayout"] = True

    logarithmic = np.log(fitnesses)

    # Pandas dataframe
    data = pd.DataFrame({f'Gen {i}': logarithmic[i] for i in range(len(logarithmic))})

    # Plot the dataframe
    ax = data[[f'Gen {i}' for i in range(len(logarithmic))]].plot(kind='box', title='boxplot')

    # Display the plot
    plt.savefig('imgs/fitnesses-fgm.png')

if __name__=='__main__':
    """ PARAMETERS """
    # taken from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4699269/table/T but with population cloud rather than single point
    global n, a, Q, s0, r, sigma, e, m, N, mu, generations
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
    N = 5 # number of individuals in population
    mu = 10 ** -3 # mutation rate

    def fitness(phenotype): # fitness function from paper (input is distance from fitness peak)
        return np.exp(-1 * a * (np.linalg.norm(phenotype) ** Q))

    generations = 3 # number of generations to run simulation for
    """ PARAMETERS """

    """ SIMULATION """
    # initialize population
    global population
    population = np.random.normal(1.5, 0.5, size=(N, n)) # random population
    global fitnesses
    fitnesses = np.zeros((generations, N))
    j = 0
    while j < fitnesses.shape[1]:
        fitnesses[0][j] = fitness(np.linalg.norm(population[j])) # calculate fitness of each individual
        j += 1

    for i in range(generations):
        next_gen()
        j = 0
        while j < fitnesses.shape[1]:
            fitnesses[i][j] = fitness(np.linalg.norm(population[j])) # calculate fitness of each individual in next generation
            j += 1
        # fitnesses[i] *= 2

    print(fitnesses)
    plot_fitnesses()

