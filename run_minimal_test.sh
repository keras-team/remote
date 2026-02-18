#!/bin/bash
set -e

PROJECT="keras-team-gcp"
ZONE="us-central1-a"
CLUSTER="pathways-minimal"
LWS_URL="https://github.com/kubernetes-sigs/lws/releases/download/v0.5.1/manifests.yaml"

echo "Waiting for cluster $CLUSTER to be RUNNING..."
while true; do
    STATUS=$(gcloud container clusters list --filter="name=$CLUSTER" --zone=$ZONE --format="value(status)")
    echo "Current status: $STATUS"
    if [[ "$STATUS" == "RUNNING" ]]; then
        break
    fi
    sleep 30
done

echo "Cluster is RUNNING. Configuring kubectl..."
# gcloud container clusters get-credentials $CLUSTER --zone=$ZONE --project=$PROJECT

echo "Installing LeaderWorkerSet controller..."
kubectl apply --server-side -f $LWS_URL

echo " waiting for LWS controller to be ready (10s)..."
sleep 10

echo "Running Pathways Example..."
export KERAS_REMOTE_PROJECT=$PROJECT
export KERAS_REMOTE_ZONE=$ZONE
export KERAS_REMOTE_CLUSTER=$CLUSTER
# Force gcloud auth (just in case) or use existing
python3 examples/pathways_example.py
