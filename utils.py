import numpy as np

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
    if len(matrix.shape) > 1:
        trans = np.array(matrix).T
    else:
        trans = matrix
    # initialize data dictionary
    data = {}
    # iterate over rows of transpose and add each row to dictionary
    for i in range(len(trans)):
        lbl = label + '_' + str(i)
        data[lbl] = trans[i]
    return data


def trim_matrices(container, flat):
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
    trimmed = []
    if flat:
        # find minimum size of any array
        min_size = min(len(v) for v in container)
        for v in container:
            if len(v) >= min_size:
                tv = v[:min_size]
                trimmed.append(tv)
        return trimmed
    # Find the minimum number of columns of the inner matrices
    min_cols = min(m.shape[1] for m in container)    
    for m in container:
        if m.shape[1] >= min_cols:
            # truncate matrix
            tm = m[:, :min_cols]
            trimmed.append(tm)
    return trimmed



def get_filename(path):
    """
    Reduces the given path to just the final filename
    and excludes the containing directories.

    Args:
        path (str): full file path

    Returns:
        (str): string of just the file name
    """
    idx = path.rindex('/') + 1
    return path[idx:]