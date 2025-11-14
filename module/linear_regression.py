import numpy as np
from .matrix_operations import (
    multiply_matrices,
    transpose_matrix,
    inverse_matrix
)

def linear_regression(X, y):
    # mengubah ke bentuk matriks
    y_col = y.reshape(-1, 1)
    # kolom intercept
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # Hitung transpose
    X_t = transpose_matrix(X_b)
    # Hitung perkalian X transpose dengan X
    XtX = multiply_matrices(X_t, X_b)
    # Inverse data
    XtX_inv = inverse_matrix(XtX)

    if isinstance(XtX_inv, str):
        # memberikan pesan error jika ternyata dataset menjadi singular ketika diinversekan
        raise ValueError(
            "Dataset tidak kompatibel: matriks dari data singular sehingga regresi linear tidak dapat dihitung."
        )

    # 4. Hitung X^T dikali y
    Xty = multiply_matrices(X_t, y_col)
    # 5. Hitung theta = (XtX)^(-1) (X^T y)
    theta = multiply_matrices(XtX_inv, Xty)

    return theta.flatten()


def predict(X, theta):
    # untuk menghitung hasil prediksi dari reg linear
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return multiply_matrices(X_b, theta.reshape(-1, 1)).flatten()


def calculate_mse(y_actual, y_predicted):
    # menghitung mse/error dari perhitungan
    return np.mean((y_actual - y_predicted) ** 2)


def fit_and_predict(X_original, y):
    # membuat hasil model/prediksi dari reg linear dan error
    theta = linear_regression(X_original, y)
    y_pred = predict(X_original, theta)
    mse = calculate_mse(y, y_pred)
    return theta, y_pred, mse
