def create_plot(X_for_plot, y_actual, beta_intercept_star, beta_slope):
    """Membuat plot Matplotlib dan mengembalikannya sebagai Base64 string."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_for_plot, y_actual, label='Data Aktual')
    X_sorted =np.sort(X_for_plot)

    #Persamaan Regresi Linear untuk visualisasi (di bidang 2D)
    y_regresi_line = beta_intercept_star + beta_slope * X_sorted

    # Plot Garis Regresi: y = beta[0] + beta[1]*x
    ax.plot(X_sorted, y_regresi_line, color='red',
            label=f'Regresi: y = {beta_intercept_star:.2f} + {beta_slope:.2f}x')
    ax.set_xlabel('Variabel X')
    ax.set_ylabel('Variabel Y')
    ax.set_title('Regresi Linear Sederhana (Representasi 2D)')
    ax.legend()

    # Konversi ke Base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"data:image/png;base64,{data}"
