import numpy as np

def add_matrices(A, B):
    return np.add(A, B)

def subtract_matrices(A, B):
    return np.subtract(A, B)

def multiply_matrices(A, B):
    return np.dot(A, B)

def transpose_matrix(A):
    return np.transpose(A)

def inverse_matrix(A):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return "Matriks tidak dapat diinvers (singular)."
