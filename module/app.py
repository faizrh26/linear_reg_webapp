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
    try:
        # 1. Ambil File dan Metadata
        uploaded_file = request.files['csv_file']

        # Ambil indeks kolom dari input form
        x_col_index = int(request.form['x_col'])
        y_col_index = int(request.form['y_col'])

        if uploaded_file.filename == '':
            return redirect(url_for('index')) # Kembali jika tidak ada file

        # 2. Panggil fungsi utilitas untuk membaca data dari stream
        # Disini file_stream akan menjadi uploaded_file.stream
        X_original, y_processed = read_matrix_from_file(
            uploaded_file, x_col_index, y_col_index
        )

        # Pisahkan X_original untuk plotting (tanpa kolom intercept)
        X_original = X_with_intercept[:, 1].reshape(-1, 1)

    except Exception as e:
        # Tangkap semua error terkait I/O dan input
        return render_template('result.html', error=f"Kesalahan Input/File: {e}")

    # 3. Panggil Modul Kalkulasi (Logika ini tetap sama)
    try:
        beta, y_pred, mse = fit_and_predict(X_original, y_processed)

    except Exception as e:
        return render_template('result.html', error=f"Error Modul Kalkulasi: {e}")

    #Visualisasi dan export csv
    plot_url = create_plot(X_original.flatten(), y_processed, y_pred, y_beta)

    export_data = {
        'X_input': X_original.flatten().tolist(),
        'Y_actual': y_processed.tolist(),
        'Y_Predicted': y_pred.tolist(),
    }

    export_success = export_to_csv(export_data, filename='data/regresi_output.csv')

    #Tampilkan Hasil
    return render_template('result.html',
                           plot_url=plot_url,
                           beta_intercept=beta[0],
                           beta_slope=beta[1],
                           mse=mse,
                           export_status=export_success,
                           rumus_regresi=f"y = {beta[0]:.4f} + {beta[1]:.4f}x")
if __name__ == '__main__':
    app.run(debug=Tree)
