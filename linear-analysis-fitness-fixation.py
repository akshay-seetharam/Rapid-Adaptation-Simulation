from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def linear_model(x, a, b):
    return a * x + b

def logistic(x, a, b, c, d):
    return a / (d + np.exp(-b * (x - c)))

def linear_analysis():
    pass

if __name__=='__main__':

    """PARAMETERS"""
    runs = 30
    """PARAMETERS"""

    data = np.genfromtxt('x-vs-Pfix.csv', delimiter=',').transpose()
    print(data)

    stdevs = [np.sqrt(x * (1 - x) / runs) + 10 ** -4 for x in data[1]]

    popt, pcov = curve_fit(linear_model, data[0], data[1])#, sigma=stdevs)
    popt2, pcov2 = curve_fit(linear_model, data[0], data[1], sigma=stdevs)
    x = np.linspace(data[0][0], data[0][-1], 1000)
    y = [linear_model(x_, *popt) for x_ in x]
    y2 = [linear_model(x_, *popt2) for x_ in x]
    plt.plot(x, y, label='Linear Model', color='orange')
    plt.plot(x, y2, label='Weighted Linear Model', color='green', linestyle='dashed')

    plt.errorbar(data[0], data[1], yerr=stdevs, fmt="o")
    plt.legend()

    plt.xlabel("Probability of Fixation")
    plt.ylabel("Impact from Beneficial Mutation")
    plt.savefig("imgs/LinearModel.png")
    plt.clf()

    residuals = [linear_model(x_, *popt) - y_ for x_, y_ in zip(data[0], data[1])]
    plt.scatter(data[0], residuals)
    plt.plot(np.linspace(0, 0.1, 1000), [0] * 1000, color='black')
    plt.title("Residuals of Linear Model")
    plt.savefig("imgs/Residuals.png")