FROM python:3.11-slim

WORKDIR /app

# Cài system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để cache layer
COPY requirements-deploy.txt .

# Cài PyTorch CPU-only (nhẹ hơn nhiều, không cần GPU trên server)
RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Cài các thư viện còn lại
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]