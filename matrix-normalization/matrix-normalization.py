import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    matrix = np.array(matrix)
    result = 0
    

    if matrix.ndim != 2:
        return None
    if axis is not None and axis >= matrix.ndim:
        return None
    
    if norm_type == 'l2':
        result = np.sqrt(np.sum(np.power(matrix, 2), axis=axis, keepdims=True))
    elif norm_type == 'l1':
        result = np.sum(np.abs(matrix), axis=axis, keepdims=True)
    elif norm_type == 'max':
        result = np.max(np.abs(matrix), axis=axis, keepdims=True)
    else:
        return None

    return np.where(result == 0, 0, matrix/result)