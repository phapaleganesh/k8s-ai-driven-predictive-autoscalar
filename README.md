# AI-Driven Predictive Horizontal Pod Autoscaler (pHPA)

This repository demonstrates a production-grade, AI-driven predictive autoscaler for Kubernetes. The solution uses Python, Prometheus, Docker, and Kubernetes best practices. The prediction model leverages Facebook Prophet for robust time-series forecasting.

## Typical Use Case

Dynamic autoscaling of microservices in Kubernetes clusters, especially when built-in Horizontal Pod Autoscaler (HPA) is insufficient or when predictive scaling is needed.

## Prerequisites

* Kubernetes cluster with kubectl access
* Prometheus deployed and scraping pod metrics
* container runtime installed
* Python 3.8+
* The deployment to scale

## Prediction Service

The Script `predictive_autoscalar.py` script automatically scale a Kubernetes deployment up or down based on predicted CPU usage, using time series forecasting.

### How It Works

#### * Configuration via Environment Variables

The script reads settings like the Prometheus server URL, deployment name, namespace, CPU target per pod, min/max replicas, and prediction intervals from environment variables (with sensible defaults).

#### * Fetching CPU Metrics

It queries Prometheus for recent CPU usage metrics for the specified deployment’s pods.
Data is retrieved for a configurable window (default: last 180 minutes).

#### * Forecasting Future CPU Usage

Uses the Prophet time series forecasting library to predict CPU usage for the next few minutes (default: 5 minutes ahead).

#### * Calculating Desired Replicas

Based on the predicted CPU usage and the target CPU per pod, it calculates the optimal number of replicas.
Ensures the replica count stays within the configured min/max bounds.

#### * Scaling the Deployment

Uses kubectl scale to adjust the number of replicas in the Kubernetes deployment.

#### * Looping

Repeats the process every 5 minutes, continuously monitoring and scaling as needed.

## Containerizaing the Predictive Service

To build predictive service image, run the following docker build commands:
```
docker build -t predictive-autoscalar:<tag> .
```

## K8s Manifest

The manifest folder containers k8s deployment file which is require for this implementation.

**prometheus** folder contains promentheus deployments file. To apply this, go to the folder and run:
```
kubectl apply -f .
```
**test-app** folder contains a test app which will be use for predictive auto scaling. To deploy this, go to the folder and run:
```
kubectl apply -f .
```
**predictive-autoscalar** folder contains predictive autoscalar. Before applying this k8s manifests, edit the ENV variables in the `predictive-autoscalar.yaml` file according to you application and require values.
To deploy this, run:
```
kubectl apply -f .
```

## Test the Predictive Autoscalar

Generate the load on deployed test app and see the logs for predictive autoscalar and sideby watch the number of pods increasing/decreasing based on CPU prediction.

## Tuning:
Adjust `TARGET_CPU_PER_POD, MIN_REPLICAS, MAX_REPLICAS` and other parameters as needed.

## Summary

This script predicts future CPU load for a Kubernetes deployment and automatically scales the number of pods to optimize resource usage and performance, using Prometheus for monitoring and Prophet for forecasting.
