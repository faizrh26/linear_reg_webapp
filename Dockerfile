# linear_reg_webapp/Dockerfile

# 1. Pilih base image
# Kita menggunakan Python versi 3.12 (sesuai venv Anda) yang ringan (slim)
FROM python:3.12-slim

# 2. Set working directory di dalam container
WORKDIR /app

# 3. Instal dependensi sistem yang diperlukan untuk matplotlib (font, library C)
# Pemasangan ini mencegah error saat Matplotlib mencoba membuat plot
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    # Font dan dependency Matplotlib
    pkg-config \
    libfreetype6-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements file dan instal dependensi Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy seluruh kode aplikasi (termasuk modul/, templates/, static/)
# KECUALI yang ada di .dockerignore
COPY . .

# 6. Set variabel lingkungan untuk Flask
# Tentukan entry point dari aplikasi Anda (module/app.py)
ENV FLASK_APP="module.app"
ENV FLASK_RUN_HOST="0.0.0.0"

# 7. Expose port yang digunakan Flask
EXPOSE 5000

# 8. Perintah untuk menjalankan aplikasi
# Gunakan 'flask run'
CMD ["flask", "run"]
