"""
Multitask Compressive Sensing: Shihao Ji, David Dunson†, and Lawrence Carin

https://www.researchgate.net/publication/224514217_Multitask_Compressive_Sensing
"""

import stan
import numpy as np

import importlib
import os
import sys
from tqdm import tqdm


# adds the parent directory to the sys.path list to import utils module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) + "/pursuit"
sys.path.append(parent_dir)


# checks that input is a numpy array
def check_numpy_array(input_data):
    if not isinstance(input_data, np.ndarray):
        raise TypeError("Input must be a NumPy array.")


# normalizes a matrix
def normalize(matrix):
    col_norms = np.linalg.norm(matrix, axis=0)
    normalized_matrix = matrix / col_norms
    return normalized_matrix


# reads the BPS stan model
with open("./bayes/mtcs_full.stan", "r") as file:
    bps_stan = file.read()


def process_results_mean(results, M, n):
    """
    Processes results from the Bayesian model

    :param results: posterior samples from STAN
    :type results: Model object
    :param M: number of signals
    :type results: int
    :param n: orignal dimension of each signal
    :type n: int

    :return: sample mean for each signal
    :rtype: dict
    """

    def create_theta_dictionary(M, n):
        theta_dict = {}
        for i in range(1, M + 1):
            key = f"Theta.{i}"
            theta_dict[key] = np.zeros(n)
        return theta_dict

    df = results.to_frame().describe().T
    theta_dict = create_theta_dictionary(M, n)
    for i in range(1, n * M + 1):
        key_dict = f"Theta.{(i - 1) % M + 1}"
        key_df = f"Theta.{(i - 1) % M + 1}.{(i - 1) % n + 1}"
        theta_dict[key_dict][(i - 1) % n] = df.at[key_df, "mean"]
    return theta_dict


def vector_to_diagonal_matrix(vector):
    if vector.ndim != 2 or vector.shape[1] != 1:
        raise ValueError("Input vector must be of shape (mx1)")

    m = vector.shape[0]
    diagonal_matrix = np.diag(
        np.squeeze(vector)
    )  # Squeezing to remove the second dimension

    return diagonal_matrix


class MT_CS_full:
    """
    Performs multitask compressive sensing with full bayesian computation

    :param X: matrix containing signals as columns
    :type X: numpy array
    :param dictionary: dictionary containing atoms
    :type dictionary: numpy array
    :param measure_matrix: measure matrix
    :type measure_matrix: numpy array
    :param proportion: percentage of each measured signal using
        the appropriate matrix
    :type proportion: int
    """

    def fit(self, X, dictionary="dct", measure_matrix="gaussian", proportion=0.5):
        # X must be a numpy array
        check_numpy_array(X)

        # imports matrix_utils module
        try:
            imported_module = importlib.import_module("matrix_utils")
            print(f"Successfully imported utils module.")

        except ModuleNotFoundError:
            print(
                f"Module '{dictionary}' not found. Please check the module name and try again."
            )

        # imports the dictionary
        try:
            generate_dict = getattr(imported_module, f"{dictionary}_dictionary")

        except NameError:
            print(f"{dictionary} not found. Please check the name and try again.")

        # imports the measure matrix
        try:
            generate_measure = getattr(imported_module, f"{measure_matrix}_matrix")
        except NameError:
            print(f"{measure_matrix} not found. Please check the name and try again.")

        # normalizes signals
        Z = normalize(X.copy().T)

        # M: number of signals, n: original dimension of each signal
        M, n = Z.shape

        # measures dimension
        m = int(n * proportion)

        # measures matrix
        Phi = normalize(generate_measure(m, n))

        # measures vectors
        Z_measure = Z @ Phi.T

        # sampling from STAN model
        signals_data = {"M": M, "n": n, "m": m, "X": Z_measure, "phi": Phi}
        posterior = stan.build(bps_stan, data=signals_data)
        fit_results = posterior.sample(num_chains=4, num_samples=1)
        results = process_results_mean(fit_results, M, n)
        return results
