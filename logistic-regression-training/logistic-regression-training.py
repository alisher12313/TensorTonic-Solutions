import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X, y = np.array(X), np.array(y)
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(steps):
        z = np.dot(X, w) + b
        y_pred = _sigmoid(z)

        error = y_pred - y

        dw = (1/n_samples) * (np.dot(X.T, error))
        db = (1/n_samples) * np.sum(error)

        w -= lr * dw
        b -= lr * db

    return w, b