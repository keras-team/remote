#!/bin/bash

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  keras-remote Cleanup Script"
echo "=========================================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}ERROR: gcloud CLI not found${NC}"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
if [ -z "$KERAS_REMOTE_PROJECT" ]; then
    echo -e "${YELLOW}KERAS_REMOTE_PROJECT not set${NC}"
    echo "Please enter your GCP project ID:"
    read -r PROJECT_ID
else
    PROJECT_ID="$KERAS_REMOTE_PROJECT"
fi

# Derive region and AR location from KERAS_REMOTE_ZONE
ZONE="${KERAS_REMOTE_ZONE:-us-central1-a}"
REGION="${ZONE%-*}"
AR_LOCATION="${REGION%%-*}"

echo -e "${YELLOW}WARNING: This will delete ALL keras-remote resources in project: $PROJECT_ID${NC}"
echo ""
echo "This includes:"
echo "  - Cloud Storage buckets (jobs and builds)"
echo "  - Artifact Registry repositories and images"
echo "  - Running Vertex AI jobs"
echo "  - GKE clusters (matching 'keras-remote-*' pattern)"
echo "  - TPU VMs (if any)"
echo ""
echo -n "Are you sure you want to continue? (yes/no): "
read -r CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Starting cleanup..."
echo ""

# Delete storage buckets
echo "1. Deleting Cloud Storage buckets..."
if gcloud storage rm -r "gs://${PROJECT_ID}-keras-remote-jobs" 2>/dev/null; then
    echo -e "   ${GREEN}✓ Deleted jobs bucket${NC}"
else
    echo -e "   ${YELLOW}ℹ Jobs bucket not found or already deleted${NC}"
fi

if gcloud storage rm -r "gs://${PROJECT_ID}-keras-remote-builds" 2>/dev/null; then
    echo -e "   ${GREEN}✓ Deleted builds bucket${NC}"
else
    echo -e "   ${YELLOW}ℹ Builds bucket not found or already deleted${NC}"
fi
echo ""

# Cancel and list Vertex AI jobs
echo "2. Checking for running Vertex AI jobs..."
RUNNING_JOBS=$(gcloud ai custom-jobs list \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --filter="state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING" \
    --format="value(name)" 2>/dev/null || echo "")

if [ -n "$RUNNING_JOBS" ]; then
    echo "   Found running jobs. Cancelling..."
    echo "$RUNNING_JOBS" | while read -r job; do
        JOB_ID=$(basename "$job")
        echo -n "   Cancelling job: $JOB_ID... "
        if gcloud ai custom-jobs cancel "$JOB_ID" --region="$REGION" --project="$PROJECT_ID" --quiet 2>/dev/null; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC}"
        fi
    done
else
    echo -e "   ${GREEN}✓ No running jobs found${NC}"
fi
echo ""

# Delete Artifact Registry repository
echo "3. Deleting Artifact Registry repository..."
if gcloud artifacts repositories describe keras-remote \
    --location="$AR_LOCATION" \
    --project="$PROJECT_ID" &> /dev/null; then
    echo -n "   Deleting keras-remote repository... "
    if gcloud artifacts repositories delete keras-remote \
        --location="$AR_LOCATION" \
        --project="$PROJECT_ID" \
        --quiet 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
    fi
else
    echo -e "   ${YELLOW}ℹ Repository not found or already deleted${NC}"
fi
echo ""

# Check for TPU VMs
echo "4. Checking for TPU VMs..."
TPU_VMS=$(gcloud compute tpus list --project="$PROJECT_ID" --format="value(name,zone)" 2>/dev/null || echo "")

if [ -n "$TPU_VMS" ]; then
    echo "   Found TPU VMs:"
    echo "$TPU_VMS" | while read -r vm zone; do
        if [ -n "$vm" ] && [ -n "$zone" ]; then
            echo -n "   Deleting $vm in $zone... "
            if gcloud compute tpus delete "$vm" --zone="$zone" --project="$PROJECT_ID" --quiet 2>/dev/null; then
                echo -e "${GREEN}✓${NC}"
            else
                echo -e "${RED}✗ Failed${NC}"
            fi
        fi
    done
else
    echo -e "   ${GREEN}✓ No TPU VMs found${NC}"
fi
echo ""

# Delete GKE clusters matching keras-remote-* pattern
echo "5. Checking for GKE clusters (matching 'keras-remote-*' pattern)..."
GKE_CLUSTERS=$(gcloud container clusters list \
    --project="$PROJECT_ID" \
    --filter="name~^keras-remote-" \
    --format="value(name,location)" 2>/dev/null || echo "")

if [ -n "$GKE_CLUSTERS" ]; then
    echo "   Found GKE clusters:"
    echo "$GKE_CLUSTERS" | while read -r cluster location; do
        if [ -n "$cluster" ] && [ -n "$location" ]; then
            echo -n "   Deleting $cluster in $location... "
            if gcloud container clusters delete "$cluster" \
                --location="$location" \
                --project="$PROJECT_ID" \
                --quiet 2>/dev/null; then
                echo -e "${GREEN}✓${NC}"
            else
                echo -e "${RED}✗ Failed${NC}"
            fi
        fi
    done
else
    echo -e "   ${GREEN}✓ No GKE clusters found${NC}"
fi
echo ""

# Check for Compute Engine VMs (in case any were created)
echo "6. Checking for Compute Engine VMs (matching 'remote-*' pattern)..."
COMPUTE_VMS=$(gcloud compute instances list \
    --project="$PROJECT_ID" \
    --filter="name~^remote-.*" \
    --format="value(name,zone)" 2>/dev/null || echo "")

if [ -n "$COMPUTE_VMS" ]; then
    echo "   Found VMs:"
    echo "$COMPUTE_VMS" | while read -r vm zone; do
        if [ -n "$vm" ] && [ -n "$zone" ]; then
            echo -n "   Deleting $vm in $zone... "
            if gcloud compute instances delete "$vm" --zone="$zone" --project="$PROJECT_ID" --quiet 2>/dev/null; then
                echo -e "${GREEN}✓${NC}"
            else
                echo -e "${RED}✗ Failed${NC}"
            fi
        fi
    done
else
    echo -e "   ${GREEN}✓ No Compute Engine VMs found${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}Cleanup complete!${NC}"
echo "=========================================="
echo ""
echo "Remaining billable resources to check manually:"
echo "  1. Any other custom Vertex AI jobs:"
echo "     https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=$PROJECT_ID"
echo ""
echo "  2. GKE clusters (if any remain):"
echo "     https://console.cloud.google.com/kubernetes/list?project=$PROJECT_ID"
echo ""
echo "  3. Cloud Build history (informational only, no cost when not building):"
echo "     https://console.cloud.google.com/cloud-build/builds?project=$PROJECT_ID"
echo ""
echo "  4. Overall billing:"
echo "     https://console.cloud.google.com/billing?project=$PROJECT_ID"
echo ""
