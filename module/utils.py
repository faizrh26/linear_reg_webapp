import numpy as np
import pandas as pd
import os
from io import StringIO

def read_matrix_from_text(X_data_str, y_data_str):
    """
    Membaca data X dan Y yang dipisahkan koma dari string input web (app.py).

    Args:
        X_data_str (str): String data X (fitur), dipisahkan koma.
        y_data_str (str): String data Y (target), dipisahkan koma.

    Returns:
        tuple: (X_matrix_with_intercept, y_vector) sebagai array NumPy.
    """
    try:
        # Konversi string ke list float
        X_list = [float(x.strip()) for x in X_data_str.split(',')]
        y_list = [float(y.strip()) for y in y_data_str.split(',')]

        # Validasi panjang data
        if len(X_list) != len(y_list):
            raise ValueError("Jumlah data X dan Y harus sama.")

        X_original = np.array(X_list).reshape(-1, 1)
        y_vector = np.array(y_list)

        # Tambahkan kolom intercept (array berisi 1) untuk keperluan regresi matriks
        X_matrix_with_intercept = np.hstack([np.ones(X_original.shape), X_original]) 

        return X_matrix_with_intercept, y_vector

    except ValueError as e:
        # Re-raise error dengan pesan yang lebih spesifik
        raise ValueError(f"Kesalahan format data: {e}")
    except Exception as e:
        raise Exception(f"Kesalahan saat memproses input string: {e}")
