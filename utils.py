import numpy as np
import os
import sys

import importlib
from tqdm import tqdm

# adds the parent directory to the sys.path list to import utils module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) + "/pursuit"
sys.path.append(parent_dir)


# checks that input is a numpy array
def check_numpy_array(input_data):
    if not isinstance(input_data, np.ndarray):
        raise TypeError("Input must be a NumPy array.")


# creates a diagonal matrix
def vector_to_diagonal_matrix(vector):
    if vector.ndim != 2 or vector.shape[1] != 1:
        raise ValueError("Input vector must be of shape (mx1)")

    m = vector.shape[0]
    diagonal_matrix = np.diag(
        np.squeeze(vector)
    )  # Squeezing to remove the second dimension

    return diagonal_matrix


# compute relative error
def compute_relative_error(V1_true, V2_approx):
    return np.linalg.norm(V1_true - V2_approx) / np.linalg.norm(V1_true)


# normalizes a matrix
def normalize(matrix):
    col_norms = np.linalg.norm(matrix, axis=0)
    normalized_matrix = matrix / col_norms
    return normalized_matrix
