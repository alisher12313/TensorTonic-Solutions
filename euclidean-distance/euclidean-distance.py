import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x, y = np.array(x), np.array(y)
    result = np.sqrt(np.sum(np.power(x - y, 2)))
    return float(result)