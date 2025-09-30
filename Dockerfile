# --- Stage 1: Build ---
FROM docker.io/python:3.11-slim AS builder
LABEL maintainer="gphapale@deloitte.com"
WORKDIR /app

# Install only what we need in the builder and download kubectl
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    wget https://dl.k8s.io/release/v1.29.0/bin/linux/arm64/kubectl --no-check-certificate -O /app/kubectl && \
    chmod +x /app/kubectl && \
    apt-get purge -y --auto-remove wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM docker.io/python:3.11-slim

WORKDIR /app
# bring kubectl and installed python packages from builder
COPY --from=builder /app/kubectl /usr/local/bin/kubectl
COPY --from=builder /install /usr/local

# install supervisor and curl (minimal), clean up apt caches
RUN apt-get update && \
    apt-get install -y --no-install-recommends supervisor curl && \
    rm -rf /var/lib/apt/lists/*

# Copy app sources and supervisor configuration
COPY predictive_autoscaler.py .
COPY requirements.txt .
COPY dashboard /app/dashboard
# place supervisord.conf in the conventional location and use it at runtime
COPY supervisord.conf /etc/supervisor/supervisord.conf

# Ensure /usr/local/bin is on PATH so kubectl and installed scripts are available
ENV PYTHONPATH=/usr/local
ENV PATH=/usr/local/bin:${PATH}

# Expose dashboard and metrics ports
EXPOSE 8080 8000

# Start supervisord with the provided config
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
