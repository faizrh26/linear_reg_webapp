import pandas as pd

def export_to_csv(data_dict, filename='data/regresi_results.csv'):
    """
    Mengekspor data hasil regresi ke file CSV.
    Input: data_dict (Dictionary data), filename (Jalur file output)
    """
    try:
        # Membuat DataFrame dari dictionary.
        # Asumsi semua list di data_dict memiliki panjang yang sama.
        df = pd.DataFrame(data_dict)

        # Simpan ke CSV di folder 'data/'
        df.to_csv(filename, index=False)
        return True
    except Exception as e:
        # Biasanya terjadi jika jalur file salah atau data_dict kosong
        print(f"ERROR: Gagal mengekspor data ke CSV. {e}")
        return False
