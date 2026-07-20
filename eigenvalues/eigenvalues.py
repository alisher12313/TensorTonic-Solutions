import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    if not isinstance(matrix, (list, np.ndarray)) or len(matrix) == 0:
        return None
    
    if not all(isinstance(row, (list, np.ndarray)) for row in matrix):
        return None
    
    if not all(len(row) == len(matrix[0]) for row in matrix):
        return None
    
    matrix = np.array(matrix, dtype=float)
    
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    
    eigenvalues = np.linalg.eigvals(matrix)
    
    sorted_indices = np.lexsort((eigenvalues.imag, eigenvalues.real))
    sorted_eigenvalues = eigenvalues[sorted_indices]
    
    return sorted_eigenvalues