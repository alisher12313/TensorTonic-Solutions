import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    X, y = np.array(X, dtype=float), np.array(y, dtype=float)
    
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)
    
    XtX_det = np.linalg.det(XtX)
    if XtX_det == 0:
        return None  
    
    XtX_inverse = np.linalg.inv(XtX)
    w = np.dot(XtX_inverse, Xty)
    print(w.ndim)
    return w