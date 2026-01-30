import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X = np.array(X)
    if X.ndim != 2:
        return None
    n = X.shape[0]
    if n <= 1:
        return None
    mu = np.mean(X, axis=0)
    X_center = X - mu
    return (X_center.T @ X_center) / (n - 1)