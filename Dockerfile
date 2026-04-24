FROM python:3.11-slim

# System deps for OpenCV and PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces expects port 7860
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    TORCH_HOME=/tmp/torch_cache \
    MPLCONFIGDIR=/tmp/mpl_cache

WORKDIR /app

# Install Python deps (CPU-only PyTorch keeps the image small enough for HF free tier)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 torchvision==0.17.2 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py /app/server.py
COPY src /app/src
COPY configs /app/configs
COPY static /app/static
COPY results /app/results

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
