FROM python:3.11-slim

# Mencegah Python menulis file .pyc ke disk
ENV PYTHONDONTWRITEBYTECODE 1
# Memastikan stdout/stderr dikirim ke terminal tanpa di-buffer
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install dependency sistem operasi (untuk OpenCV dan pustaka dasar)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependency Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode proyek
COPY . .

# Ekspos port Uvicorn
EXPOSE 8000

# Jalankan server FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
