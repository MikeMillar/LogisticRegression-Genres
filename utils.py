import numpy as np
from sklearn.decomposition import PCA

def aggregate_stats(matrix):
    """
    Takes a 2D matrix of values, flattens it, and
    collects statistics such as sum, mean, standard
    deviation, minimum value, and maximum values.

    Args:
        matrix ([[float]]): 2D matrix of float values

    Returns:
        (float): Summation of all matrix values
        (float): Mean of all matrix values
        (float): Standard deviation of the matrix values
        (float): The minimum value in the matrix
        (float): The maximum value in the matrix
    """
    flat = matrix.flatten()
    return np.sum(flat), np.mean(flat), np.std(flat), np.min(flat), np.max(flat)

def pca_reduction(matrix, dimensions):
    """
    Takes a 2D matrix of values and reduces it to
    a one-dimensional matrix of values using 
    principal component analysis (PCA).

    Args:
        matrix ([[float]]): 2D matrix of float values

    Returns:
        ([float]): A vector of the PCA reduction
    """
    pca = PCA(n_components=dimensions)
    return pca.fit_transform(matrix).flatten()