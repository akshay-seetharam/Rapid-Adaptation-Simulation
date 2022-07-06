# noinspection PyClassHasNoInit
class FitnessFunctions:
    def univariate_basic(cls, fitness: float) -> float:
        """returns input"""
        return fitness

    def univariate_converge(cls, fitness: float) -> float:
        """fitness function where 0.1 is the optimal fitness"""
        return 1 - (fitness - 0.1) ** 2

    def nvariate_converge(cls, phenotype: list) -> float:
        return 1 - FitnessFunctions.nvariate_average(FitnessFunctions, phenotype) ** 2

    def nvariate_average(cls, phenotype: list) -> float:
        return sum(phenotype) / len(phenotype)