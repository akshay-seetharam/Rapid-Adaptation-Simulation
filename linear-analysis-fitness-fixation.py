from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def linear_model(x, a, b):
    return a * x + b

def logistic_model(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

def cubic_model(x, a, b, c):
    return a * x**2 + b * x + c

def cubic_analysis(data, stdevs):
    popt, pcov = curve_fit(cubic_model, data[0], data[1])
    popt2, pcov2 = curve_fit(cubic_model, data[0], data[1], sigma = stdevs)

    x = np.linspace(data[0][0], data[0][-1], 1000)
    y = [cubic_model(x_, *popt) for x_ in x]
    y2 = [cubic_model(x_, *popt2) for x_ in x]
    plt.plot(x, y, label='Cubic Model', color='orange')
    plt.plot(x, y2, label='Weighted Cubic Model', color='green', linestyle='dashed')

    plt.errorbar(data[0], data[1], yerr=stdevs, fmt="o")
    plt.legend()

    plt.xlabel("Probability of Fixation")
    plt.ylabel("Impact from Beneficial Mutation")
    plt.savefig("imgs/CubicModel.png")
    plt.clf()

    residuals = [cubic_model(x_, *popt) - y_ for x_, y_ in zip(data[0], data[1])]
    plt.scatter(data[0], residuals)
    plt.plot(np.linspace(0, 0.1, 1000), [0] * 1000, color='black')
    plt.title("Residuals of Cubic Model")
    plt.savefig("imgs/CubicResiduals.png")
    plt.clf()

def logistic_analysis(data, stdevs):
    popt, pcov = curve_fit(logistic_model, data[0], data[1])
    popt2, pcov2 = curve_fit(logistic_model, data[0], data[1], sigma = stdevs)

    x = np.linspace(data[0][0], data[0][-1], 1000)
    y = [logistic_model(x_, *popt) for x_ in x]
    y2 = [logistic_model(x_, *popt2) for x_ in x]
    plt.plot(x, y, label='Logistic Model', color='orange')
    plt.plot(x, y2, label='Weighted Logistic Model', color='green', linestyle='dashed')

    plt.errorbar(data[0], data[1], yerr=stdevs, fmt="o")
    plt.legend()

    plt.xlabel("Probability of Fixation")
    plt.ylabel("Impact from Beneficial Mutation")
    plt.savefig("imgs/LogisticModel.png")
    plt.clf()

    residuals = [logistic_model(x_, *popt) - y_ for x_, y_ in zip(data[0], data[1])]
    plt.scatter(data[0], residuals)
    plt.plot(np.linspace(0, 0.1, 1000), [0] * 1000, color='black')
    plt.title("Residuals of Logistic Model")
    plt.savefig("imgs/LogisticResiduals.png")
    plt.clf()

def linear_analysis(data, stdevs):
    popt, pcov = curve_fit(linear_model, data[0], data[1])
    popt2, pcov2 = curve_fit(linear_model, data[0], data[1], sigma = stdevs)

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
    plt.savefig("imgs/LinearResiduals.png")
    plt.clf()

if __name__=='__main__':

    """PARAMETERS"""
    runs = 30
    """PARAMETERS"""

    data = np.genfromtxt('x-vs-Pfix.csv', delimiter=',').transpose()
    i = 0
    while i < len(data[1]):
        data[1][i] = np.log(1 / (1 - data[1][i]))
        i += 1
    print(data)

    stdevs = [np.sqrt(x * (1 - x) / runs) + 10 ** -4 for x in data[1]]

    plt.style.use('dark_background')

    linear_analysis(data, stdevs)
    # logistic_analysis(data, stdevs)
    # cubic_analysis(data, stdevs)