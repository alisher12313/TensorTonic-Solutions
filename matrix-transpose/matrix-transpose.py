import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    rows, cols = A.shape
    A_transpose = np.zeros((cols, rows))

    for row in range(rows):
        for col in range(cols):
            A_transpose[col, row] = A[row, col]
            
    return A_transpose
