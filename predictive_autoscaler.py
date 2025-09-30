import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
import subprocess
import time
import threading
import logging
from typing import Optional
from prometheus_client import start_http_server, Gauge, Counter
import json
import tempfile
import os

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus-server.default.svc.cluster.local:9090")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME", "my-app")
NAMESPACE = os.getenv("NAMESPACE", "default")
TARGET_CPU_PER_POD = float(os.getenv("TARGET_CPU_PER_POD", "0.2"))
MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "1"))
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "10"))
PREDICT_MINUTES_AHEAD = int(os.getenv("PREDICT_MINUTES_AHEAD", "5"))
SCRAPE_MINUTES = int(os.getenv("SCRAPE_MINUTES", "180"))

# safety and runtime options
STEP = os.getenv("PROM_STEP", "15s")  # use Prometheus-compatible step like '15s' or '1m'
DRY_RUN = os.getenv("DRY_RUN", "false").lower() in ("1", "true", "yes")
USE_K8S_CLIENT = os.getenv("USE_K8S_CLIENT", "false").lower() in ("1", "true", "yes")
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "300"))
MAX_REPLICA_DELTA = int(os.getenv("MAX_REPLICA_DELTA", "2"))
MIN_POINTS_FOR_PREDICTION = int(os.getenv("MIN_POINTS_FOR_PREDICTION", "30"))

# Reactive scaling config (short checks between prediction runs)
REACTIVE_CHECK_ENABLED = os.getenv("REACTIVE_CHECK_ENABLED", "false").lower() in ("1", "true", "yes")
REACTIVE_CHECK_INTERVAL_SECONDS = int(os.getenv("REACTIVE_CHECK_INTERVAL_SECONDS", "30"))
REACTIVE_WINDOW_SECONDS = int(os.getenv("REACTIVE_WINDOW_SECONDS", "60"))
REACTIVE_USE_PER_POD_AVG = os.getenv("REACTIVE_USE_PER_POD_AVG", "true").lower() in ("1", "true", "yes")

logging.basicConfig(level=logging.INFO)

# model cache to avoid retraining every loop when data hasn't changed
_MODEL = None
_MODEL_TRAINED_UPTO: Optional[datetime] = None

# Shared state for scaling (protected by lock)
last_scaled_ts = None
last_replicas = None
_state_lock = threading.Lock()

# Prometheus metrics
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
g_predicted_cpu = Gauge('predictive_autoscaler_predicted_cpu', 'Predicted total CPU cores')
g_desired_replicas = Gauge('predictive_autoscaler_desired_replicas', 'Desired replica count from prediction')
c_scale_actions = Counter('predictive_autoscaler_scale_actions_total', 'Number of scale actions executed')


def fetch_cpu_metrics():
    end = datetime.utcnow()
    start = end - timedelta(minutes=SCRAPE_MINUTES)
    # total CPU across pods (used for prediction training)
    query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",pod=~"{DEPLOYMENT_NAME}.*"}}[5m]))'
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={
                "query": query,
                "start": int(start.timestamp()),
                "end": int(end.timestamp()),
                "step": STEP,
            },
            timeout=20,
        )
    except requests.RequestException as e:
        logging.error(f"Prometheus request failed: {e}")
        return None

    if resp.status_code != 200:
        logging.error(f"Prometheus returned status {resp.status_code}: {resp.text[:200]}")
        return None

    try:
        payload = resp.json()
    except ValueError:
        logging.error("Prometheus response not JSON")
        return None

    result = payload.get("data", {}).get("result", [])
    if not result:
        logging.warning("Prometheus returned empty result")
        return None

    values = result[0].get("values") or result[0].get("value")
    if not values:
        logging.warning("Prometheus result has no values")
        return None

    # values expected to be list of [timestamp, value]
    df = pd.DataFrame(values, columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"], unit="s")
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)

    # resample to uniform frequency derived from STEP for Prophet
    try:
        freq = None
        if isinstance(STEP, str) and STEP.endswith('s'):
            freq = f"{int(STEP[:-1])}S"
        elif isinstance(STEP, str) and STEP.endswith('m'):
            freq = f"{int(STEP[:-1])}T"
        else:
            freq = STEP
        df = df.set_index('ds').resample(freq).mean().interpolate().reset_index()
    except Exception:
        logging.debug("Failed to resample; using raw samples")

    return df

def predict_cpu(df):
    global _MODEL, _MODEL_TRAINED_UPTO

    if df is None or len(df) < MIN_POINTS_FOR_PREDICTION:
        logging.warning(f"Not enough data points for prediction: {0 if df is None else len(df)} < {MIN_POINTS_FOR_PREDICTION}")
        return None

    # retrain model only if data has newer last timestamp than previous train
    last_ts = df['ds'].max()
    if _MODEL is None or _MODEL_TRAINED_UPTO is None or last_ts > _MODEL_TRAINED_UPTO:
        logging.info("Training new Prophet model")
        model = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
        try:
            model.fit(df.rename(columns={"ds": "ds", "y": "y"}))
            _MODEL = model
            _MODEL_TRAINED_UPTO = last_ts
        except Exception as e:
            logging.error(f"Prophet fit failed: {e}")
            return None

    try:
        future = _MODEL.make_future_dataframe(periods=PREDICT_MINUTES_AHEAD, freq="min")
        forecast = _MODEL.predict(future)
        predicted_cpu = float(forecast.iloc[-1]["yhat"])
        predicted_cpu = max(predicted_cpu, 0.0)
        try:
            g_predicted_cpu.set(predicted_cpu)
        except Exception:
            logging.debug('Failed to set predicted CPU metric')
        # write the predicted future series to a local JSON file so the dashboard can
        # visualize the predicted CPU over time. We take the last PREDICT_MINUTES_AHEAD
        # rows (they correspond to the future horizon) and write as [[ts_seconds, value], ...]
        try:
            future_forecast = forecast.tail(PREDICT_MINUTES_AHEAD)
            series = []
            for _, row in future_forecast.iterrows():
                ts = int(pd.to_datetime(row['ds']).timestamp())
                val = float(row['yhat'])
                series.append([ts, val])
            tmpf = None
            outpath = os.getenv('PREDICTED_SERIES_PATH', '/tmp/predicted_series.json')
            try:
                fd, tmpf = tempfile.mkstemp(prefix='pred_series_', dir=os.path.dirname(outpath))
                with os.fdopen(fd, 'w') as fh:
                    json.dump({'series': series}, fh)
                # atomic replace
                os.replace(tmpf, outpath)
            except Exception:
                logging.debug('Failed to write predicted series file')
                try:
                    if tmpf and os.path.exists(tmpf):
                        os.unlink(tmpf)
                except Exception:
                    pass
        except Exception:
            logging.debug('Failed to prepare predicted series')
        return predicted_cpu
    except Exception as e:
        logging.error(f"Prophet predict failed: {e}")
        return None

def _compute_desired_replicas_from_cpu(predicted_cpu: float) -> int:
    return int(max(MIN_REPLICAS, min(MAX_REPLICAS, round(predicted_cpu / TARGET_CPU_PER_POD))))


def _get_current_replicas() -> Optional[int]:
    # try kubernetes client first if enabled
    if USE_K8S_CLIENT:
        try:
            from kubernetes import client, config
            config.load_incluster_config()
            apps = client.AppsV1Api()
            dep = apps.read_namespaced_deployment(DEPLOYMENT_NAME, NAMESPACE)
            return int(dep.spec.replicas)
        except Exception as e:
            logging.warning(f"k8s client couldn't read deployment: {e}")

    # fallback to kubectl
    try:
        out = subprocess.check_output([
            "kubectl", "get", "deployment", DEPLOYMENT_NAME, "-n", NAMESPACE,
            "-o", "jsonpath={.spec.replicas}"
        ], text=True)
        return int(out.strip())
    except Exception as e:
        logging.warning(f"kubectl get deployment failed: {e}")
        return None


def scale_deployment(predicted_cpu: float, last_scaled_ts: Optional[float], last_replicas: Optional[int]):
    if predicted_cpu is None:
        logging.warning("No prediction available; skipping scale")
        return None, last_scaled_ts

    desired = _compute_desired_replicas_from_cpu(predicted_cpu)
    cur = _get_current_replicas()
    if cur is None:
        cur = last_replicas

    # enforce max per-iteration delta
    if cur is not None:
        delta = desired - cur
        if abs(delta) > MAX_REPLICA_DELTA:
            desired = cur + (MAX_REPLICA_DELTA if delta > 0 else -MAX_REPLICA_DELTA)
            desired = int(max(MIN_REPLICAS, min(MAX_REPLICAS, desired)))

    now = time.time()
    if DRY_RUN:
        logging.info(f"DRY_RUN: would scale from {cur} to {desired} replicas (predicted_cpu={predicted_cpu:.4f})")
        try:
            g_desired_replicas.set(desired)
        except Exception:
            logging.debug('Failed to set desired replicas metric')
        return desired, now

    if last_scaled_ts and (now - last_scaled_ts) < COOLDOWN_SECONDS:
        logging.info(f"Cooldown active ({int(now-last_scaled_ts)}s elapsed) - skipping scale to {desired}")
        return None, last_scaled_ts

    # try k8s client scale
    if USE_K8S_CLIENT:
        try:
            from kubernetes import client, config
            config.load_incluster_config()
            apps = client.AppsV1Api()
            body = {"spec": {"replicas": desired}}
            apps.patch_namespaced_deployment_scale(DEPLOYMENT_NAME, NAMESPACE, body)
            logging.info(f"Scaled deployment via k8s client to {desired} replicas")
            try:
                g_desired_replicas.set(desired)
                c_scale_actions.inc()
            except Exception:
                logging.debug('Failed to update scale metrics')
            return desired, now
        except Exception as e:
            logging.warning(f"k8s client scale failed: {e}; falling back to kubectl")

    # fallback to kubectl
    try:
        subprocess.run([
            "kubectl", "scale", f"deployment/{DEPLOYMENT_NAME}", f"--replicas={desired}", "-n", NAMESPACE
        ], check=True)
        logging.info(f"Scaled deployment to {desired} replicas via kubectl")
        try:
            g_desired_replicas.set(desired)
            c_scale_actions.inc()
        except Exception:
            logging.debug('Failed to update scale metrics')
        return desired, now
    except subprocess.CalledProcessError as e:
        logging.error(f"kubectl scale failed: {e}")
        return None, last_scaled_ts


def _instant_total_cpu():
    """Query Prometheus for an instant total CPU usage across pods (uses 1m rate window).
    Returns float cores or None on error.
    """
    # use configured reactive window for the rate() range vector (e.g. '60s')
    q = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",pod=~"{DEPLOYMENT_NAME}.*"}}[{REACTIVE_WINDOW_SECONDS}s]))'
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": q}, timeout=8)
        if r.status_code != 200:
            logging.debug(f"instant CPU query bad status {r.status_code}")
            return None
        js = r.json()
        res = js.get('data', {}).get('result', [])
        if not res:
            return 0.0
        # take first result value
        v = res[0].get('value')
        if not v or len(v) < 2:
            return None
        return float(v[1])
    except Exception as e:
        logging.debug(f"instant cpu query failed: {e}")
        return None


def reactive_worker():
    """Background worker that performs short-interval reactive checks and triggers
    a bounded scale-up if recent total CPU indicates a sudden spike.
    """
    global last_scaled_ts, last_replicas
    logging.info(f"Reactive worker started: enabled={REACTIVE_CHECK_ENABLED}, interval={REACTIVE_CHECK_INTERVAL_SECONDS}s")
    while REACTIVE_CHECK_ENABLED:
        try:
            cpu = _instant_total_cpu()
            if cpu is None:
                time.sleep(REACTIVE_CHECK_INTERVAL_SECONDS)
                continue

            # compute desired replicas based on configured mode
            cur = _get_current_replicas()
            if cur is None:
                # will reuse last known replicas if available
                with _state_lock:
                    cur = last_replicas

            if REACTIVE_USE_PER_POD_AVG:
                # Instead of using total CPU, query per-pod average CPU
                try:
                    q = f'avg(sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",pod=~"{DEPLOYMENT_NAME}.*"}}[{REACTIVE_WINDOW_SECONDS}s])))'
                    r = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": q}, timeout=8)
                    if r.status_code == 200:
                        js = r.json()
                        res = js.get('data', {}).get('result', [])
                        if res:
                            v = res[0].get('value')
                            if v and len(v) >= 2:
                                avg_per_pod = float(v[1])
                            else:
                                avg_per_pod = None
                        else:
                            avg_per_pod = None
                    else:
                        avg_per_pod = None
                except Exception:
                    avg_per_pod = None

                if avg_per_pod is None:
                    time.sleep(REACTIVE_CHECK_INTERVAL_SECONDS)
                    continue

                # If average per-pod CPU is higher than target, scale proportionally
                # desired_replicas = ceil(current_replicas * (avg_per_pod / TARGET_CPU_PER_POD))
                try:
                    import math
                    if cur is None:
                        # if we don't know current count, estimate 1
                        cur = 1
                    scale_factor = max(1.0, avg_per_pod / TARGET_CPU_PER_POD)
                    desired = int(max(MIN_REPLICAS, min(MAX_REPLICAS, math.ceil(cur * scale_factor))))
                except Exception:
                    desired = None
            else:
                # compute desired replicas from *total* CPU observed now
                desired = _compute_desired_replicas_from_cpu(cpu)

            with _state_lock:
                cur_last_scaled = last_scaled_ts
                cur_last_replicas = last_replicas

            # only attempt reactive scale-up (don't scale-down here)
            cur = _get_current_replicas()
            if cur is None:
                cur = cur_last_replicas

            if cur is None:
                logging.debug('Reactive check: current replica count unknown; skipping')
                time.sleep(REACTIVE_CHECK_INTERVAL_SECONDS)
                continue

            if desired is None:
                time.sleep(REACTIVE_CHECK_INTERVAL_SECONDS)
                continue

            if desired <= cur:
                # no need to scale up
                time.sleep(REACTIVE_CHECK_INTERVAL_SECONDS)
                continue

            # enforce MAX_REPLICA_DELTA
            delta = desired - cur
            if delta > MAX_REPLICA_DELTA:
                desired = cur + MAX_REPLICA_DELTA
                desired = int(max(MIN_REPLICAS, min(MAX_REPLICAS, desired)))

            # call scale_deployment. When using per-pod avg mode we already computed
            # the desired replica count, so pass a synthetic "predicted_cpu" that
            # causes scale_deployment to calculate the same desired value.
            if REACTIVE_USE_PER_POD_AVG and 'desired' in locals() and desired is not None:
                predicted_for_scaling = float(desired) * TARGET_CPU_PER_POD
            else:
                predicted_for_scaling = cpu
            scaled, ts = scale_deployment(predicted_for_scaling, cur_last_scaled, cur_last_replicas)
            if scaled is not None:
                with _state_lock:
                    last_replicas = scaled
                    last_scaled_ts = ts
        except Exception:
            logging.exception('Reactive worker encountered error')
        time.sleep(REACTIVE_CHECK_INTERVAL_SECONDS)

def main():
    global last_scaled_ts, last_replicas
    last_scaled_ts = None
    last_replicas = None
    iterations = 0
    # start reactive worker thread if enabled
    if REACTIVE_CHECK_ENABLED:
        t = threading.Thread(target=reactive_worker, daemon=True)
        t.start()
    # start metrics server
    try:
        start_http_server(METRICS_PORT)
        logging.info(f"Started metrics HTTP server on :{METRICS_PORT}")
    except Exception as e:
        logging.warning(f"Failed to start metrics HTTP server: {e}")
    while True:
        iterations += 1
        try:
            df = fetch_cpu_metrics()
            if df is not None and len(df) >= MIN_POINTS_FOR_PREDICTION:
                predicted_cpu = predict_cpu(df)
                if predicted_cpu is not None:
                    logging.info(f"Iteration {iterations}: predicted total CPU={predicted_cpu:.4f}")
                    scaled, ts = scale_deployment(predicted_cpu, last_scaled_ts, last_replicas)
                    if scaled is not None:
                        with _state_lock:
                            last_replicas = scaled
                            last_scaled_ts = ts
                else:
                    logging.warning("Prediction failed; skipping scaling this iteration")
            else:
                logging.warning(f"Insufficient data for prediction. Points={0 if df is None else len(df)}")
        except Exception as e:
            logging.exception(f"Error in autoscaler loop: {e}")
        time.sleep(180)  # 3 minutes

if __name__ == "__main__":
    main()
