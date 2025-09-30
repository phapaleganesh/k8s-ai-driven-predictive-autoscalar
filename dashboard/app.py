"""Flask dashboard for predictive_autoscaler.

Serves a simple SPA that polls the autoscaler's Prometheus metrics endpoint
and Prometheus server to display current/predicted CPU, replicas, and a chart.
"""
from flask import Flask, render_template, jsonify, request
import os
import requests
import logging
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='static', template_folder='templates')

PROMETHEUS_URL = os.getenv('PROMETHEUS_URL', 'http://prometheus-server.default.svc.cluster.local:9090')
METRICS_ENDPOINT = os.getenv('AUTOSCALER_METRICS_URL', 'http://localhost:8000')
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'my-app')
NAMESPACE = os.getenv('NAMESPACE', 'default')

logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/overview')
def overview():
    # Read metrics from autoscaler metrics endpoint
    try:
        r = requests.get(f"{METRICS_ENDPOINT}/metrics", timeout=5)
        content = r.text
    except Exception as e:
        logging.warning(f"Failed to read local metrics: {e}")
        content = ''

    # parse simple values from text (prometheus exposition format)
    def parse_metric(name):
        for line in content.splitlines():
            if line.startswith(name + ' '):
                try:
                    return float(line.split(' ')[1])
                except Exception:
                    return None
        return None

    predicted = parse_metric('predictive_autoscaler_predicted_cpu')
    desired = parse_metric('predictive_autoscaler_desired_replicas')
    # fallback to None
    return jsonify({
        'deployment': DEPLOYMENT_NAME,
        'namespace': NAMESPACE,
        'predicted_cpu': predicted,
        'desired_replicas': desired,
        'target_cpu': float(os.getenv('TARGET_CPU_PER_POD', '0.2')),
        'pred_minutes': int(os.getenv('PREDICT_MINUTES_AHEAD', '5')),
        'scrape_minutes': int(os.getenv('SCRAPE_MINUTES', '180')),
        'metrics_endpoint': METRICS_ENDPOINT,
    })

@app.route('/api/timeseries')
def timeseries():
    # Query Prometheus for recent CPU usage and predicted metric (we rely on predicted cpu metric emitted by autoscaler)
    end = datetime.utcnow()
    start = end - timedelta(minutes=60)
    step = request.args.get('step', '15s')
    # compute per-pod CPU (sum rate per pod) then avg across pods so we get "average CPU per pod"
    q_cpu = f'avg(sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{NAMESPACE}",pod=~"{DEPLOYMENT_NAME}.*"}}[5m])))'
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params={
            'query': q_cpu,
            'start': int(start.timestamp()),
            'end': int(end.timestamp()),
            'step': step,
        }, timeout=10)
        data = r.json().get('data', {}).get('result', [])
        if data:
            values = data[0].get('values', [])
            cpu_series = [[int(float(t)), float(v)] for t, v in values]
        else:
            cpu_series = []
    except Exception as e:
        logging.warning(f"Prometheus timeseries fetch failed: {e}")
        cpu_series = []

    # also fetch predicted metric from metrics endpoint (if exposed)
    try:
        m = requests.get(f"{METRICS_ENDPOINT}/metrics", timeout=5).text
        pred_val = None
        for line in m.splitlines():
            if line.startswith('predictive_autoscaler_predicted_cpu '):
                try:
                    pred_val = float(line.split(' ')[1])
                except Exception:
                    pred_val = None
                break
    except Exception:
        pred_val = None

    # also attempt to read a predicted series file written by the autoscaler
    predicted_series = None
    try:
        path = os.getenv('PREDICTED_SERIES_PATH', '/tmp/predicted_series.json')
        if os.path.exists(path):
            with open(path, 'r') as fh:
                import json
                j = json.load(fh)
                predicted_series = j.get('series')
    except Exception:
        predicted_series = None

    return jsonify({'cpu': cpu_series, 'predicted_cpu': pred_val, 'predicted_series': predicted_series})


@app.route('/api/raw_metrics')
def raw_metrics():
    """Return raw metrics text from the autoscaler's metrics endpoint for debugging."""
    try:
        r = requests.get(f"{METRICS_ENDPOINT}/metrics", timeout=5)
        return (r.text, r.status_code, {'Content-Type': 'text/plain; charset=utf-8'})
    except Exception as e:
        logging.warning(f"Failed to fetch raw metrics: {e}")
        return ("", 500, {'Content-Type': 'text/plain; charset=utf-8'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('DASHBOARD_PORT', '8080')))
