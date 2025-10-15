from flask import Flask, render_template, request
import numpy as np
import io, base64
import matplotlib.pyplot as plt

# Import modul terkait matriks dan regresi linear
# Note: modul masih dalam development sehingga kode masih bisa berubah2
from .linear_regression import fit_and_predict
from .data_handler import export_to_csv
from .utils import read_matrix_from_file

app = Flask(__name__, template_folder='../templates')

def create_plot(X, y, y_pred, beta):
    """Membuat plot Matplotlib dan mengembalikannya sebagai Base64 string."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, label='Data Aktual')

    # Plot Garis Regresi: y = beta[0] + beta[1]*x
    ax.plot(X, y_pred, color='red', 
            label=f'Regresi: y = {beta[0]:.2f} + {beta[1]:.2f}x')

    ax.set_xlabel('Variabel X')
    ax.set_ylabel('Variabel Y')
    ax.set_title('Regresi Linear Sederhana')
    ax.legend()

    # Konversi ke Base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig) 
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"data:image/png;base64,{data}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    """Endpoint POST: Menerima data, kalkulasi, visualisasi, dan menampilkan hasil."""
    # 1. AMBIL DATA FORM DENGAN SAFETY CHECK
    # Gunakan .get() untuk menghindari KeyError
    uploaded_file = request.files.get('csv_file')
    x_cols_str = request.form.get('x_cols')
    y_col_str = request.form.get('y_col')

    # Cek apakah input form kosong
    if not x_cols_str or not y_col_str or uploaded_file is None or uploaded_file.filename == '':
         return render_template('result.html', error="Kesalahan Input/File: Semua input tidak boleh kosong dan file harus diunggah.")

    # 2. KONVERSI DATA DAN I/O
    try:
        # Konversi ke integer
        y_col_index = int(y_col_str)
        x_col_indices = [int(i.strip()) for i in x_cols_str.split(',')]

        # Panggil fungsi utilitas
        X_original, y_processed = read_matrix_from_file(
            uploaded_file, x_col_indices, y_col_index
        )

    except ValueError as e:
        # Menangkap error konversi (y_col atau x_cols bukan angka)
        return render_template('result.html', error=f"Kesalahan Input/File: Indeks kolom harus angka yang valid. ({e})")

    except Exception as e:
        # Tangkap semua error lain (I/O, File tidak valid, dll.)
        return render_template('result.html', error=f"Kesalahan I/O: {e}")

    # 3. Panggil Modul Kalkulasi
    try:
        # X_original dan y_processed PASTI ada di sini jika tidak ada return di atas
        beta, y_pred, mse = fit_and_predict(X_original, y_processed)

    except Exception as e:
        # JALUR KEGAGALAN KALKULASI: HARUS ADA RETURN
        return render_template('result.html', error=f"Error Modul Kalkulasi: {e}")

    # 4. Visualisasi dan export csv (JALUR SUKSES)
    # NOTE: Saat Multiple Regression, X_original adalah (N, >1) dimensi. 
    # create_plot hanya bisa menerima 1D array. Kita pakai fitur pertama (kolom X1) untuk plotting
    plot_url = create_plot(X_original[:, 0].flatten(), y_processed, y_pred, beta)

    export_data = {
        'X_input_Fitur_Pertama': X_original[:, 0].flatten().tolist(),
        'Y_actual': y_processed.tolist(),
        'Y_Predicted': y_pred.tolist(),
    }

    # Tambahkan semua fitur X ke export_data
    for i in range(X_original.shape[1]):
        export_data[f'X_Fitur_{i+1}'] = X_original[:, i].flatten().tolist()

    export_success = export_to_csv(export_data, filename='data/regresi_output.csv')

    # 5. Tampilkan Hasil (RETURN WAJIB)
    # Untuk regresi berganda, kita tidak bisa hanya menampilkan beta[1].
    # Kita harus menampilkan semua koefisien:
    koefisien_str = f"y = {beta[0]:.4f} (Intercept) "
    for i in range(1, len(beta)):
         koefisien_str += f"+ {beta[i]:.4f}x_{i} "

    return render_template('result.html',
                            plot_url=plot_url,
                            beta_intercept=beta[0],
                            beta_slope=beta[1], # Masih bisa ditampilkan, tapi hanya untuk fitur pertama
                            mse=mse,
                            export_status=export_success,
                            rumus_regresi=koefisien_str) # <--- INI ADALAH RETURN AKHIR YANG HILANG
if __name__ == '__main__':
    app.run(debug=Tree)
