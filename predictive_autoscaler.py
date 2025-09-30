import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
import subprocess
import time
import logging

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-server.default.svc.cluster.local:9090")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "my-app")
NAMESPACE = os.getenv("NAMESPACE", "default")
TARGET_CPU_PER_POD = float(os.getenv("TARGET_CPU_PER_POD", "0.2"))
MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "10"))
PREDICT_MINUTES_AHEAD = int(os.getenv("PREDICT_MINUTES_AHEAD", "5"))
SCRAPE_MINUTES = int(os.getenv("SCRAPE_MINUTES", "180"))

logging.basicConfig(level=logging.INFO)

def fetch_cpu_metrics():
    end = datetime.utcnow()
    start = end - timedelta(minutes=SCRAPE_MINUTES)
    query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",pod=~"{DEPLOYMENT_NAME}.*"}}[5m]))'
    response = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": "15"
        }
    )
    results = response.json()["data"]["result"]
    if not results:
        logging.warning("No data from Prometheus.")
        return None
    data = results[0]["values"]
    df = pd.DataFrame(data, columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"], unit="s")
    df["y"] = df["y"].astype(float)
    return df

def predict_cpu(df):
    model = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=PREDICT_MINUTES_AHEAD, freq="min")
    forecast = model.predict(future)
    predicted_cpu = forecast.iloc[-1]["yhat"]
    return max(predicted_cpu, 0)

def scale_deployment(predicted_cpu):
    desired_replicas = int(max(MIN_REPLICAS, min(MAX_REPLICAS, round(predicted_cpu / TARGET_CPU_PER_POD))))
    subprocess.run([
        "kubectl", "scale", f"deployment/{DEPLOYMENT_NAME}",
        f"--replicas={desired_replicas}", "-n", NAMESPACE
    ], check=True)
    logging.info(f"Scaled to {desired_replicas} replicas based on predicted CPU: {predicted_cpu:.4f}")

def main():
    while True:
        try:
            df = fetch_cpu_metrics()
            if df is not None and len(df) > 10:
                predicted_cpu = predict_cpu(df)
                scale_deployment(predicted_cpu)
            else:
                logging.warning("Insufficient data for prediction.")
        except Exception as e:
            logging.error(f"Error in autoscaler loop: {e}")
        time.sleep(180)  # 3 minutes

if __name__ == "__main__":
    main()
