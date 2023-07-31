import numpy as np
from matplotlib import pyplot as plt
import random

class PopulationGenome():
    def __init__(self, population, polymorphic_sites, S_b, activated_proportion, mutation_prob, num_recombinations=None):
        self.population = population
        self.polymorphic_sites = polymorphic_sites
        self.S_b = S_b
        self.activated_proportion = activated_proportion
        self.mutation_prob = mutation_prob
        self.num_recombinations = int(mutation_prob * population) if num_recombinations is None else num_recombinations

        self.genome = np.zeros((population, polymorphic_sites))

        for i in range(population):
            genome[i] = np.array([np.random.binomial(1, activated_proportion) for _ in range(polymorphic_sites)])

        self.fitness_deltas = []
        self.fitness_progression = []

    def average_fitness(self):
        return np.average(np.sum(self.genome, axis=1) * self.S_b)

    def horizontal_gene_transfer(self):
        # source retains allele, recipient receives allele from source
        new_genome = self.genome.copy()
        for _ in range(self.num_recombinations):
            donor = random.choice((0, genome.shape[0] - 1))
            locus = random.choice((0, genome.shape[1] - 1))
            recipient = donor
            while recipient == donor:
                recipient = random.choice((0, genome.shape[0] - 1))
            new_genome[recipient][locus] = new_genome[donor][locus]

        return new_genome

    def reproduce(self):
        new_genome = np.zeros_like(self.genome)
        weights = np.exp(np.sum(self.genome, axis=1))
        weights /= np.sum(weights)
        for i in range(new_genome.shape[0]):
            new_genome[i] = self.genome[np.random.choice(self.genome.shape[0], p=weights)]
            # weighted average with exponential of fitness
        num_mutants = int(self.population * self.mutation_prob)
        for _ in range(num_mutants):
            print('Mutating')
            new_polymorphic_site = np.zeros((self.population, 1))
            new_polymorphic_site[np.random.choice(population)][0] = 1
            new_genome = np.concatenate((new_genome, new_polymorphic_site), axis=1)

        new_genome = np.delete(new_genome, np.argwhere(np.all(new_genome[..., :] == 0, axis=0)), axis=1)
        # stop tracking loci that are no longer polymorphic

        return new_genome

    def reproductive_update(self):
        genome = horizontal_gene_transfer(self.genome)
        genome = reproduce(genome)
        self.genome = genome

    def simulate(self, generations, plot=True):
        for gen in range(generations):
            old_fitness = self.average_fitness()
            self.reproductive_update()
            new_fitness = self.average_fitness()
            self.fitness_progression.append(new_fitness)
            self.fitness_deltas.append(new_fitness - old_fitness)
            print(f'Done with generation {gen}')

        plt.figure(figsize=(60, 60))
        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].plot(range(generations), self.fitness_deltas, color='red')
        ax[0].set_xlabel('Generation')
        ax[0].set_ylabel('Delta Fitness')
        ax[0].set_title('Change in Fitness by Generation')

        ax[1].plot(range(generations), self.fitness_progression, color='red')
        ax[1].set_xlabel('Generation')
        ax[1].set_ylabel('Fitness')
        ax[1].set_title('Fitness by Generation')

        plt.savefig(f'{self.num_recombinations} recombinations.png')

        print(self.genome, self.genome.shape)


if __name__=='__main__':
    print('Hello Transfer Three World')
    #TODO modifying recombination rate with epistasis
    #TODO environmental pressure?
    #TODO cmoparison to meiotic theory
    #TODO read paper from Good & Ferrare https://www.biorxiv.org/content/10.1101/2023.07.12.548717v1.full.pdf

    num_recombinations_list = [0, 1, 10, 100, 1000]
    for i in num_recombinations_list:
        population = new PopulationGenome(10**4, 1, 0.01, 0.01, 10**-3, i)
        population.simulate()