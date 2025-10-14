import numpy as np

import numpy as np

def linear_regression(X, y):
    """
    Fungsi ASLI Teman Anda: Menghitung theta (koefisien).
    Catatan: Fungsi ini menambahkan intercept (X_b) sendiri.
    """
    # X_b = np.c_[np.ones((X.shape[0], 1)), X] <--- Baris ini dihapus/diganti
    # Karena app.py sekarang TIDAK mengirim intercept, gunakan logika asli:
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # Rumus Normal Equation
    theta = np.linalg.solve(X_b.T.dot(X_b), X_b.T.dot(y))
    return theta

def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    return X_b.dot(theta)


def calculate_mse(y_actual, y_predicted):
    """Menghitung Mean Squared Error (MSE)."""
    return np.mean((y_actual - y_predicted)**2)

# --- Fungsi lengkap ---
def fit_and_predict(X_original, y):
    """
    Menggunakan fungsi diatas untuk fit dan prediksi, lalu menghitung MSE.
    Args:
        X_original (np.array): Data fitur TANPA kolom intercept (X_1).
        y (np.array): Data target.
    Returns:
        theta, y_pred, mse
    """
    # FIT
    theta = linear_regression(X_original, y)
    # PREDICT
    y_pred = predict(X_original, theta)
    # METRIK
    mse = calculate_mse(y, y_pred)
    # Kembalikan Koefisien (Beta), Prediksi, dan MSE
    return theta, y_pred, mse
