def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X, y = np.array(X), np.array(y)
    I = np.eye(X.shape[1])
    penatly = I * lam 
    return np.linalg.inv(X.T @ X + penatly) @ X.T @ y