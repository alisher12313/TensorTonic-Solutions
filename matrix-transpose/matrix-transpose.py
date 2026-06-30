import numpy as np

def matrix_transpose(A):
    A = np.array(A)
    row, column = A.shape
    result = np.zeros((column, row), dtype=A.dtype)

    for i in range(row):
        for j in range(column):
            result[j, i] = A[i, j]

    return result
