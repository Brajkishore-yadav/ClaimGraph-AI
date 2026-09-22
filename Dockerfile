FROM python:3.9-slim

WORKDIR /app

# Install system build dependencies and libomp for XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libomp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-generate synthetic data and run model pipeline inside build
RUN python scripts/generate_data.py --customers 2000 --policies 3000 --claims 4000 --seed 42
RUN python scripts/run_pipeline.py

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
