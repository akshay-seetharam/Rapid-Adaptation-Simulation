import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from alive_progress import alive_bar as bar

def select_parents(population, fitness_func):
    """
    Select the parents that will reproduce
    """
    # Calculate the fitnesses of the population
    chances = np.ones_like((population.shape[0])) - np.log([fitness_func(i) for i in population]) # get logarithmic fitnesses
    chances /= sum(chances) # normalize the chances
    # Select parents with replacement, weighted by fitness
    parents = []
    for i in range(N):
        for j in chances:
            if j < 10**-3:
                j = 10**-3
        choice= np.random.choice(range(N), p=chances) # TODO: Check with jimmy if I should use raw fitnesses or logarithmic fitnesses or something else
        parents.append(population[choice])

    return parents

def next_gen(population, fitness_func):
    """
    Calculate the next generation of the population
    """
    # select parents that will reproduce
    parents = select_parents(population, fitness_func)
    # mutate offspring
    mutations = np.zeros_like(population)
    for i in range(len(population)):
        if np.random.rand() > mu:
            continue
        vector = np.random.random((n)) * 2 - 1 # generate a random mutation vector
        vector *= r / np.linalg.norm(vector) # scale to r
        mutations[i] = vector # assign the mutation vector to the offspring
    return mutations, parents

def plot_fitnesses(fitnesses):
    """
    Plot the fitnesses of the population
    """
    # Set the figure size
    plt.rcParams["figure.figsize"] = [12.0, 3.5]
    plt.rcParams["figure.autolayout"] = True

    logarithmic = np.log(fitnesses)

    # Pandas dataframe
    data = pd.DataFrame({f'Gen {i}': fitnesses[i] for i in range(len(logarithmic))})

    # Plot the dataframe
    ax = data[[f'Gen {i}' for i in range(len(logarithmic))]].plot(kind='box', title='boxplot')

    # Display the plot
    plt.savefig('imgs/fitnesses-fgm.png')

def simulation(n, a, Q, s0, r, sigma, x, e, m, N, mu, fitness, generations, mean, stdev):
    # initialize population
    population = np.random.normal(mean, stdev, size=(N, n)) # random population # TODO: Define mean and stdev separately for each phenotypic trait
    fitnesses = np.zeros((generations, N))
    j = 0
    while j < fitnesses.shape[1]:
        fitnesses[0][j] = fitness(np.linalg.norm(population[j])) # calculate fitness of each individual
        j += 1

    for i in range(generations):
        mutations, parents = next_gen(population, fitness)
        population = np.add(population, mutations) # mutate population
        yield
        print(population)
        j = 0
        while j < fitnesses.shape[1]:
            fitnesses[i][j] = fitness(np.linalg.norm(population[j])) # calculate fitness of each individual in next generation
            j += 1

    print(fitnesses)
    plot_fitnesses(fitnesses)

if __name__=='__main__':
    """ PARAMETERS """
    # taken from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4699269/table/T but with population cloud rather than single point
    # fitness = - Q * exp(-a * distance to peak) + s0
    n = 2 # dimension of phenotypic space
    a = 1.0 # robustness parameter in fitness function
    Q = 1.0 # epistasis parameter in fitness function
    s0 = 0.5 # maladaptaion of ancestor
    r = 0.5 # length of mutational vector
    sigma = 0.01 # standard deviation of normal deviates of the mutation vector on each axis
    def x(): # scaled distance to the optimum # TODO: Research in more detail and understand
        return r * np.sqrt(n) / (2 * d)
    e = 1.0 # epistasis defined as big long expression from table, click URL in comment above
    m = 1 # pleiotropy, number of phenotypes affected by each mutation # TODO: Make mutations affect genotype before phenotype
    N = 10**3 # number of individuals in population
    mu = 10**-3 # mutation rate

    def fitness(phenotype): # fitness function from paper (input is distance from fitness peak)
        return np.exp(-1 * a * (np.linalg.norm(phenotype) ** Q)) + s0

    generations = 20 # number of generations to run simulation for

    mean = 10 # starting phenotypic mean
    stdev = 1.5 # starting phenotypic standard deviation
    """ PARAMETERS """

    """ SIMULATION """
    with bar(generations) as b:
        for i in simulation(n, a, Q, s0, r, sigma, x, e, m, N, mu, fitness, generations, mean, stdev):
            b()