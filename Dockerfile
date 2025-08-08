# --- Stage 1: Build ---
FROM python:3.11-slim AS builder
MAINTAINER "gphapale@deloitte.com"
WORKDIR /app

RUN apt-get update && \
    apt-get install -y wget && \
    wget https://dl.k8s.io/release/v1.29.0/bin/linux/arm64/kubectl --no-check-certificate && \
    chmod +x kubectl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /app/kubectl /usr/local/bin/kubectl
COPY --from=builder /install /usr/local

COPY predictive_autoscaler.py .
COPY requirements.txt .

# Ensure environment variables for Python
ENV PYTHONPATH=/usr/local

CMD ["python", "predictive_autoscaler.py"]