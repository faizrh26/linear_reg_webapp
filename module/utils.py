import numpy as np
import pandas as pd
import os
from io import StringIO
import numpy as np
import pandas as pd

def read_matrix_from_file(file_stream, x_col_indices, y_col_index):
    """
    Membaca data fitur (X) dan target (y) dari file stream yang diunggah.

    Args:
        file_stream: Objek file yang dikirim oleh request.files.
        x_col_indices (int): Indeks kolom untuk fitur X.
        y_col_index (int): Indeks kolom untuk target y.

    Returns:
        tuple: (X_matrix_with_intercept, y_vector) sebagai array NumPy.
    """
    try:
        # Baca file stream langsung menggunakan Pandas
        df = pd.read_csv(file_stream)
        num_cols = len(df.columns)
        # Pastikan indeks yang diminta valid
        if y_col_index >= num_cols or y_col_index < 0:
            raise IndexError(f"Indeks kolom Y ({y_col_index}) di luar batas kolom.")

        for idx in x_col_indices:
            if idx >= num_cols or idx < 0:
                raise IndexError(f"Indeks kolom X ({idx}) di luar batas kolom.")

        # Ekstrak kolom X dan y berdasarkan indeks
        X_df = df.iloc[:, x_col_indices]

        y_vector = df.iloc[:, y_col_index].values.astype(float)
        X_original = X_df.values.astype(float)
        # Tambahkan kolom intercept (array berisi 1)
        # X_matrix_with_intercept = np.hstack([np.ones(X_original.shape), X_original])
        # if X_matrix_with_intercept.shape[0] != y_vector.shape[0]:
        #     raise ValueError("Jumlah baris X dan Y tidak cocok setelah pembacaan.")

        return X_original, y_vector

    except Exception as e:
        # penanganan untuk kesalahan Pandas (misal: kolom tidak ditemukan)
        if isinstance(e, IndexError):
            raise e
        raise Exception(f"Kesalahan saat memproses file CSV: {e}")

def export_to_csv(data_dict, filename='data/regresi_output.csv'):
    """
    Mengekspor hasil regresi ke file CSV. (Fungsi FINAL Tim Web)

    Args:
        data_dict (dict): Dictionary berisi data (X_Input, Y_Actual, Y_Predicted).
        filename (str): Path dan nama file output.

    Returns:
        bool: True jika sukses, False jika gagal.
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df = pd.DataFrame(data_dict)
        df.to_csv(filename, index=False)
        print(f"INFO: Data berhasil diekspor ke {filename}")
        return True
    except Exception as e:
        print(f"ERROR: Gagal mengekspor data ke CSV. {e}")
        return False
