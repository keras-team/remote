#!/bin/bash
export KERAS_REMOTE_PROJECT="keras-team-gcp"
export KERAS_REMOTE_ZONE="us-central1-a"
export KERAS_REMOTE_CLUSTER="pathways-minimal"

# Ensure kubectl is configured
echo "Configuring kubectl..."
gcloud container clusters get-credentials $KERAS_REMOTE_CLUSTER --zone $KERAS_REMOTE_ZONE --project $KERAS_REMOTE_PROJECT

# Run the example
echo "Running GKE Example..."
python3 examples/example_gke.py
