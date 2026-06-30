import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a, b = np.array(a), np.array(b)
    dot_product = np.dot(a, b)
    eucl_a = np.linalg.norm(a)
    eucl_b = np.linalg.norm(b)

    if eucl_a == 0 or eucl_b == 0:
        return 0

    return dot_product / (eucl_a * eucl_b)