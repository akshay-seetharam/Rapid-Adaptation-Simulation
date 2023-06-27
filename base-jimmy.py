import numpy as np
import numpy
from numpy.random import poisson
from math import log, exp
import sys
from numpy.random import binomial
from numpy.random import choice
from numpy.random import exponential
from scipy.special import gamma
import matplotlib.pyplot as plt


def evolve_modifier_delta(N, s1, U1, s2, U2, s1_m, U1_m, s2_m, U2_m, mu, tmax, teq, sd):
    lineages = np.asarray([[0, N]])  # first column  fitness, second column population size
    modifier_lineages = np.asarray([[0, 0]])
    X_final = 0
    Um = 0
    tfix = 0
    for t in range(0, tmax + teq):
        if t == teq:
            X_final = 0
            Um = mu

        # deterministic growth
        population_size = (np.sum(np.multiply(np.exp(lineages[:, 0]), lineages[:, 1])) + np.sum(
            np.multiply(np.exp(modifier_lineages[:, 0]), modifier_lineages[:, 1])))
        expected_sizes = np.multiply(np.exp(lineages[:, 0]), lineages[:, 1]) * (N / population_size)
        expected_modifier_sizes = np.multiply(np.exp(modifier_lineages[:, 0]), modifier_lineages[:, 1]) * (
                    N / population_size)

        # poisson sampling non-mutants
        sizes = poisson((1 - U1 - U2 - Um) * expected_sizes)
        modifier_sizes = poisson((1 - U1_m - U2_m) * expected_modifier_sizes)

        # how many mutants
        number_neutral_modifier_mutants = binomial(np.sum(expected_sizes), Um)
        number_beneficial_mutants = binomial(np.sum(expected_sizes), U1)
        number_deleterious_mutants = binomial(np.sum(expected_sizes), U2)
        number_beneficial_modifier_mutants = binomial(np.sum(expected_modifier_sizes), U1_m)
        number_deleterious_modifier_mutants = binomial(np.sum(expected_modifier_sizes), U2_m)

        # who mutates
        new_beneficial_lineages = np.add(choice(lineages[:, 0], size=number_beneficial_mutants, replace=True,
                                                p=(expected_sizes + (np.sum(expected_sizes) == 0) * np.ones(1)) / (
                                                            np.sum(expected_sizes) + (np.sum(expected_sizes) == 0))),
                                         s1 * np.ones(number_beneficial_mutants))
        # new_deleterious_lineages=np.add(choice(lineages[:,0],size=number_deleterious_mutants,replace=True,p=expected_sizes/np.sum(expected_sizes)),-1*exponential(s2,size=number_deleterious_mutants))
        new_deleterious_lineages = np.add(choice(lineages[:, 0], size=number_deleterious_mutants, replace=True,
                                                 p=(expected_sizes + (np.sum(expected_sizes) == 0) * np.ones(1)) / (
                                                             np.sum(expected_sizes) + (np.sum(expected_sizes) == 0))),
                                          s2 * np.ones(number_deleterious_mutants))
        new_neutral_modifier_lineages = np.add(
            choice(lineages[:, 0], size=number_neutral_modifier_mutants, replace=True,
                   p=(expected_sizes + (np.sum(expected_sizes) == 0) * np.ones(1)) / (
                               np.sum(expected_sizes) + (np.sum(expected_sizes) == 0))),
            sd * np.ones(number_neutral_modifier_mutants))
        new_beneficial_modifier_lineages = np.add(
            choice(modifier_lineages[:, 0], size=number_beneficial_modifier_mutants, replace=True,
                   p=(expected_modifier_sizes + (np.sum(expected_modifier_sizes) == 0) * np.ones(1)) / (
                               np.sum(expected_modifier_sizes) + (np.sum(expected_modifier_sizes) == 0))),
            s1_m * np.ones(number_beneficial_modifier_mutants))
        # new_deleterious_modifier_lineages=np.add(choice(modifier_lineages[:,0],size=number_deleterious_modifier_mutants,replace=True,p=expected_modifier_sizes/np.sum(expected_modifier_sizes)),-1*exponential(s2_m,size=number_deleterious_modifier_mutants))
        new_deleterious_modifier_lineages = np.add(
            choice(modifier_lineages[:, 0], size=number_deleterious_modifier_mutants, replace=True,
                   p=(expected_modifier_sizes + (np.sum(expected_modifier_sizes) == 0) * np.ones(1)) / (
                               np.sum(expected_modifier_sizes) + (np.sum(expected_modifier_sizes) == 0))),
            s2_m * np.ones(number_deleterious_modifier_mutants))

        new_lineages = np.hstack((np.dstack(((new_beneficial_lineages), np.ones(number_beneficial_mutants))),
                                  np.dstack(((new_deleterious_lineages), np.ones(number_deleterious_mutants)))))
        new_modifier_lineages = np.hstack((np.dstack(
            ((new_beneficial_modifier_lineages), np.ones(number_beneficial_modifier_mutants))), np.dstack(
            ((new_deleterious_modifier_lineages), np.ones(number_deleterious_modifier_mutants))), np.dstack(
            ((new_neutral_modifier_lineages), np.ones(number_neutral_modifier_mutants)))))

        # update vectors
        lineages[:, 1] = sizes
        lineages = np.delete(lineages, np.where(sizes == 0)[0], axis=0)
        modifier_lineages[:, 1] = modifier_sizes
        # if np.shape(modifier_lineages)[0]>1:
        modifier_lineages = np.delete(modifier_lineages, np.where(modifier_sizes == 0)[0],
                                      axis=0)  # this will delete array
        lineages = np.vstack((new_lineages[0], lineages))
        modifier_lineages = np.vstack((new_modifier_lineages[0], modifier_lineages))

        if len(modifier_lineages) == 0:
            modifier_lineages = np.asarray([[0, 0]])

        # remove minimum fitness
        X_bar = (np.multiply(lineages[:, 0], lineages[:, 1]).sum() + np.multiply(modifier_lineages[:, 0],
                                                                                 modifier_lineages[:, 1]).sum()) / (
                            np.sum(lineages[:, 1]) + np.sum(modifier_lineages[:, 1]))
        X_final += X_bar
        lineages[:, 0] = lineages[:, 0] - X_bar
        modifier_lineages[:, 0] = modifier_lineages[:, 0] - X_bar

        # check for when modifier fixes
        if np.sum(lineages[:, 1]) < 1:
            tfix = t - teq
            break
    v = X_final / (tmax - teq)
    return tfix, v