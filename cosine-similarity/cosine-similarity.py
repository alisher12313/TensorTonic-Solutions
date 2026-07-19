import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a, b = np.array(a), np.array(b)
    ab_product = np.dot(a, b)
    ab_eucl = eucl_norm(a) * eucl_norm(b)
    
    if ab_eucl == 0:
        return 0
    return ab_product / ab_eucl

def eucl_norm(a):
    return np.sqrt(np.sum(np.power(a, 2)))