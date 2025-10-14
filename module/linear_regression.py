import numpy as np

def linear_regression(X, y):
    # Tambahkan kolom 1 untuk bias (intercept)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # Rumus Normal Equation: (XᵀX)⁻¹ Xᵀ y
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta

def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b.dot(theta)
