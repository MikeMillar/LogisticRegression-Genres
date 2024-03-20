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



def matrix_to_columns(matrix, label):
    """
    Takes a matrix where each row represents unique data for
    a specific instance. It then transposes this matrix so that
    each row now represents a column for a specific feature of
    that data and stores each of them in a dictionary.

    Args:
        matrix ([[x]]): matrix where each row is a series of data
                        for a specific instance.
        label (string): base label name, which will have indices
                        append to it.
    
    Returns:
        (dictionary): dictionary where each key is a series of data
                      for multiple instances.
    """
    # convert to np.matrix and transpose
    trans = np.array(matrix).T
    # initialize data dictionary
    data = {}
    # iterate over rows of transpose and add each row to dictionary
    for i in range(len(trans)):
        lbl = label + '_' + str(i)
        data[lbl] = trans[i]
    return data


def trim_matrices(container, n_cols):
    """
    Takes a container of matrices, and either pads or
    truncates each matrix so that they have the desired
    number of columns.

    Args:
        container ([matrix]): A list of matrices
        n_cols (int): Number of columns each matrix should have

    Returns:
        ([matrix]): A list of matrices of same width
    """
    # Find the minimum number of columns of the inner matrices
    min_cols = min(m.shape[1] for m in container)

    
    trimmed = []
    for m in container:
        if m.shape[1] >= n_cols:
            # truncate matrix
            tm = m[:, :n_cols]
            trimmed.append(tm)
        else:
            # pad matrix
            tm = np.pad(m, ((0, 0), (0, n_cols - m.shape[1])), mode='constant')
            trimmed.append(tm)
    return trimmed



def normalize_rows(matrix):
    """
    Normalizes each row of the matrix so that the
    sum of each matrix is 1.

    Args:
        matrix ([[x]]): 2D arraylike structure

    Returns:
        ([[x]]): Normalized version of the original matrix
    """
    row_sums = np.sum(matrix, axis=1)
    return matrix / row_sums[:, np.newaxis]