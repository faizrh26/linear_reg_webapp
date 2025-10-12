from flask import Flask, render_template, request
import numpy as np
import io, base64
import matplotlib.pyplot as plt

# Import modul terkait matriks dan regresi linear
# Note: modul masih dalam development sehingga kode masih bisa berubah2
from .linear_regression import fit_and_predict
from .data_handler import export_to_csv

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

@app.route('/calculata', methods=['POST'])
def calculate():
    """Endpoint POST: Menerima data, kalkulasi, visualisasi, dan menampilkan hasil."""

    try:
        X_input_str = request.form['X_data']
        y_input_str = request_form['y_data']

        X_list = [float(x.strip()) for x in X_input_str.split(',')]
        y_list = [float(y.strip()) for y in y_input_str.split(',')]

        X_original = np.array(X_list).reshape(-1, 1) # Data asli X (untuk Plot)
        y_processed = np.array(y_list)

        X_with_intercept = np.hstack([np.ones(X_original.shape), X_original])

    except Exception as e:
        return render_template('result.html', error=f'Input tidak valid. Detail error: {e}')

    try:
        # Panggil fungsi matriks -- menunggu development selesai
        beta, y_pred, mse = fit_and_predict(X_with_intercept, y_processed)

    except Exception as e:
        return render_template('result.html', error=f"Error Modul Kalkulasi. Detail: {e}")

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
