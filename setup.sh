#!/bin/bash

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  keras-remote Setup Script"
echo "=========================================="
echo ""

# ==========================================
#  Phase 1: Gather all user input upfront
# ==========================================

SETUP_GKE=true

# Get or set project ID
if [ -z "$KERAS_REMOTE_PROJECT" ]; then
    echo ""
    echo -e "${YELLOW}KERAS_REMOTE_PROJECT not set${NC}"
    echo "Please enter your GCP project ID:"
    read -r PROJECT_ID
else
    PROJECT_ID="$KERAS_REMOTE_PROJECT"
fi

# Accelerator node pool selection (GKE only)
NODE_TYPE_CHOICE="1"
GPU_TYPE=""
GPU_MACHINE_TYPE=""
TPU_TOPOLOGY=""
TPU_MACHINE_TYPE=""
TPU_NUM_NODES=1
TPU_POOL_NAME=""

if [ "$SETUP_GKE" = true ]; then
    echo ""
    echo "What type of accelerator node pool do you want to add?"
    echo "  1) CPU only (default — already included with cluster)"
    echo "  2) GPU"
    echo "  3) TPU"
    echo ""
    echo -n "Enter your choice (1/2/3) [1]: "
    read -r NODE_TYPE_CHOICE
    NODE_TYPE_CHOICE="${NODE_TYPE_CHOICE:-1}"

    case "$NODE_TYPE_CHOICE" in
        2)
            # GPU selection
            echo ""
            echo "Select GPU type:"
            echo "  1) T4       (NVIDIA Tesla T4)"
            echo "  2) L4       (NVIDIA L4)"
            echo "  3) A100     (NVIDIA A100 40GB)"
            echo "  4) A100-80GB (NVIDIA A100 80GB)"
            echo "  5) H100     (NVIDIA H100 80GB)"
            echo ""
            echo -n "Enter your choice (1-5): "
            read -r GPU_CHOICE

            case "$GPU_CHOICE" in
                1) GPU_TYPE="nvidia-tesla-t4";  GPU_MACHINE_TYPE="n1-standard-4" ;;
                2) GPU_TYPE="nvidia-l4";         GPU_MACHINE_TYPE="g2-standard-4" ;;
                3) GPU_TYPE="nvidia-tesla-a100"; GPU_MACHINE_TYPE="a2-highgpu-1g" ;;
                4) GPU_TYPE="nvidia-a100-80gb";  GPU_MACHINE_TYPE="a2-ultragpu-1g" ;;
                5) GPU_TYPE="nvidia-h100-80gb";  GPU_MACHINE_TYPE="a3-highgpu-1g" ;;
                *)
                    echo -e "${RED}Invalid GPU choice. Skipping GPU node pool.${NC}"
                    NODE_TYPE_CHOICE="1"
                    ;;
            esac
            ;;
        3)
            # TPU selection
            echo ""
            echo "Select TPU type:"
            echo "  1) v5litepod (TPU v5 Lite Pod)"
            echo "  2) v5p       (TPU v5p)"
            echo "  3) v6e       (TPU v6e / Trillium)"
            echo "  4) v3        (TPU v3)"
            echo ""
            echo -n "Enter your choice (1-4): "
            read -r TPU_CHOICE

            case "$TPU_CHOICE" in
                1)
                    TPU_POOL_NAME="tpu-v5litepod-pool"
                    echo ""
                    echo "Select topology for v5litepod:"
                    echo "  1) 1x1 (1 chip)"
                    echo "  2) 2x2 (4 chips) (Recommended)"
                    echo "  3) 2x4 (8 chips)"
                    echo ""
                    echo -n "Enter your choice (1-3) [2]: "
                    read -r TOPO_CHOICE
                    TOPO_CHOICE="${TOPO_CHOICE:-2}"
                    case "$TOPO_CHOICE" in
                        1) TPU_TOPOLOGY="1x1"; TPU_MACHINE_TYPE="ct5lp-hightpu-1t"; TPU_NUM_NODES=1 ;;
                        2) TPU_TOPOLOGY="2x2"; TPU_MACHINE_TYPE="ct5lp-hightpu-4t"; TPU_NUM_NODES=1 ;;
                        3) TPU_TOPOLOGY="2x4"; TPU_MACHINE_TYPE="ct5lp-hightpu-8t"; TPU_NUM_NODES=1 ;;
                        *) echo -e "${RED}Invalid topology.${NC}"; NODE_TYPE_CHOICE="1" ;;
                    esac
                    ;;
                2)
                    TPU_POOL_NAME="tpu-v5p-pool"
                    echo ""
                    echo "Select topology for v5p:"
                    echo "  1) 2x2 (8 chips) (Recommended)"
                    echo "  2) 2x4 (16 chips)"
                    echo ""
                    echo -n "Enter your choice (1-2) [1]: "
                    read -r TOPO_CHOICE
                    TOPO_CHOICE="${TOPO_CHOICE:-1}"
                    case "$TOPO_CHOICE" in
                        1) TPU_TOPOLOGY="2x2"; TPU_MACHINE_TYPE="ct5p-hightpu-4t"; TPU_NUM_NODES=2 ;;
                        2) TPU_TOPOLOGY="2x4"; TPU_MACHINE_TYPE="ct5p-hightpu-4t"; TPU_NUM_NODES=4 ;;
                        *) echo -e "${RED}Invalid topology.${NC}"; NODE_TYPE_CHOICE="1" ;;
                    esac
                    ;;
                3)
                    TPU_POOL_NAME="tpu-v6e-pool"
                    echo ""
                    echo "Select topology for v6e:"
                    echo "  1) 2x2 (8 chips) (Recommended)"
                    echo "  2) 2x4 (16 chips)"
                    echo ""
                    echo -n "Enter your choice (1-2) [1]: "
                    read -r TOPO_CHOICE
                    TOPO_CHOICE="${TOPO_CHOICE:-1}"
                    case "$TOPO_CHOICE" in
                        1) TPU_TOPOLOGY="2x2"; TPU_MACHINE_TYPE="ct6e-standard-4t"; TPU_NUM_NODES=2 ;;
                        2) TPU_TOPOLOGY="2x4"; TPU_MACHINE_TYPE="ct6e-standard-4t"; TPU_NUM_NODES=4 ;;
                        *) echo -e "${RED}Invalid topology.${NC}"; NODE_TYPE_CHOICE="1" ;;
                    esac
                    ;;
                4)
                    TPU_POOL_NAME="tpu-v3-pool"
                    echo ""
                    echo "Select topology for v3:"
                    echo "  1) 2x2 (8 chips) (Recommended)"
                    echo "  2) 4x4 (32 chips)"
                    echo ""
                    echo -n "Enter your choice (1-2) [1]: "
                    read -r TOPO_CHOICE
                    TOPO_CHOICE="${TOPO_CHOICE:-1}"
                    case "$TOPO_CHOICE" in
                        1) TPU_TOPOLOGY="2x2"; TPU_MACHINE_TYPE="ct3p-hightpu-4t"; TPU_NUM_NODES=2 ;;
                        2) TPU_TOPOLOGY="4x4"; TPU_MACHINE_TYPE="ct3p-hightpu-4t"; TPU_NUM_NODES=8 ;;
                        *) echo -e "${RED}Invalid topology.${NC}"; NODE_TYPE_CHOICE="1" ;;
                    esac
                    ;;
                *)
                    echo -e "${RED}Invalid TPU choice. Skipping TPU node pool.${NC}"
                    NODE_TYPE_CHOICE="1"
                    ;;
            esac
            ;;
        1)
            ;;
        *)
            echo -e "${RED}Invalid accelerator choice. Defaulting to CPU only.${NC}"
            NODE_TYPE_CHOICE="1"
            ;;
    esac
fi

# ==========================================
#  Summary of selections
# ==========================================
echo ""
echo "=========================================="
echo "  Configuration Summary"
echo "=========================================="
echo ""
echo -e "  Project:     ${GREEN}$PROJECT_ID${NC}"
if [ "$SETUP_GKE" = true ]; then
    echo -e "  Backend:     ${GREEN}GKE${NC}"
    case "$NODE_TYPE_CHOICE" in
        2) echo -e "  Accelerator: ${GREEN}GPU ($GPU_TYPE)${NC}" ;;
        3) echo -e "  Accelerator: ${GREEN}TPU ($TPU_POOL_NAME, topology: $TPU_TOPOLOGY)${NC}" ;;
        *) echo -e "  Accelerator: ${GREEN}CPU only${NC}" ;;
    esac
fi
echo ""
echo "=========================================="
echo "  All questions answered. Starting setup..."
echo "=========================================="
echo ""

# ==========================================
#  Phase 2: Execute provisioning (no user input beyond this point)
# ==========================================

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}ERROR: gcloud CLI not found${NC}"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project ID env var if it wasn't already set
if [ -z "$KERAS_REMOTE_PROJECT" ]; then
    export KERAS_REMOTE_PROJECT="$PROJECT_ID"
    echo -e "${GREEN}Set KERAS_REMOTE_PROJECT=$PROJECT_ID${NC}"
    echo "Add this to your ~/.bashrc or ~/.zshrc:"
    echo "  export KERAS_REMOTE_PROJECT=$PROJECT_ID"
    echo ""
else
    echo -e "${GREEN}Using project: $PROJECT_ID${NC}"
    echo ""
fi

# Set default zone (optional)
if [ -z "$KERAS_REMOTE_ZONE" ]; then
    DEFAULT_ZONE="us-central1-a"
    echo -e "${YELLOW}KERAS_REMOTE_ZONE not set, using default: $DEFAULT_ZONE${NC}"
    echo "To change, add to your ~/.bashrc or ~/.zshrc:"
    echo "  export KERAS_REMOTE_ZONE=your-preferred-zone"
    echo ""
fi

# Set the project
echo "Setting gcloud project..."
gcloud config set project "$PROJECT_ID"
echo ""

# Check authentication
echo "Checking authentication..."
if ! gcloud auth application-default print-access-token &> /dev/null; then
    echo -e "${YELLOW}Application Default Credentials not found${NC}"
    echo "Running: gcloud auth application-default login"
    gcloud auth application-default login
    echo ""
else
    echo -e "${GREEN}✓ Already authenticated${NC}"
    echo ""
fi

# Enable required APIs
echo "Enabling required GCP APIs (this may take a few minutes)..."
echo ""

# Common APIs for all backends
APIS=(
    "compute.googleapis.com"           # Compute Engine API
    "cloudbuild.googleapis.com"        # Cloud Build API
    "artifactregistry.googleapis.com"  # Artifact Registry API
    "storage.googleapis.com"           # Cloud Storage API
)

# Add GKE API if selected
if [ "$SETUP_GKE" = true ]; then
    APIS+=("container.googleapis.com")         # GKE API
fi

for api in "${APIS[@]}"; do
    echo -n "  Enabling $api... "
    if gcloud services enable "$api" --project="$PROJECT_ID" 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
        echo -e "${YELLOW}You may need to enable billing for this project${NC}"
    fi
done
echo ""

# Derive region and AR location from KERAS_REMOTE_ZONE
ZONE="${KERAS_REMOTE_ZONE:-us-central1-a}"
REGION="${ZONE%-*}"
AR_LOCATION="${REGION%%-*}"

# Create Artifact Registry repository
echo "Setting up Artifact Registry repository..."
REPO_NAME="keras-remote"

if gcloud artifacts repositories describe "$REPO_NAME" \
    --location="$AR_LOCATION" \
    --project="$PROJECT_ID" &> /dev/null; then
    echo -e "${GREEN}✓ Repository already exists: $AR_LOCATION/$REPO_NAME${NC}"
else
    echo "Creating repository: $AR_LOCATION/$REPO_NAME"
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$AR_LOCATION" \
        --description="keras-remote container images" \
        --project="$PROJECT_ID"
    echo -e "${GREEN}✓ Repository created${NC}"
fi
echo ""

# Configure Docker authentication for Artifact Registry
echo "Configuring Docker authentication..."
gcloud auth configure-docker "$AR_LOCATION-docker.pkg.dev" --quiet
echo -e "${GREEN}✓ Docker authentication configured${NC}"
echo ""

# GKE Setup (if selected)
if [ "$SETUP_GKE" = true ]; then
    echo ""
    echo "=========================================="
    echo "  GKE Backend Setup"
    echo "=========================================="
    echo ""

    CLUSTER_NAME="${KERAS_REMOTE_CLUSTER:-keras-remote-cluster}"

    echo "Setting up GKE cluster: $CLUSTER_NAME in zone: $ZONE"
    echo ""

    # Check if cluster exists
    if gcloud container clusters describe "$CLUSTER_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT_ID" &> /dev/null; then
        echo -e "${GREEN}✓ GKE cluster already exists: $CLUSTER_NAME${NC}"
    else
        echo "Creating GKE cluster..."
        gcloud container clusters create "$CLUSTER_NAME" \
            --project="$PROJECT_ID" \
            --zone="$ZONE" \
            --num-nodes=1 \
            --machine-type=e2-standard-4 \
            --disk-size=50GB \
            --scopes=gke-default,storage-full \
            --no-enable-autoupgrade
        echo -e "${GREEN}✓ GKE cluster created${NC}"
    fi

    # Configure kubectl
    echo ""
    echo "Configuring kubectl access..."
    export USE_GKE_GCLOUD_AUTH_PLUGIN=True
    gcloud container clusters get-credentials "$CLUSTER_NAME" \
        --zone="$ZONE" \
        --project="$PROJECT_ID"
    echo -e "${GREEN}✓ kubectl configured${NC}"

    # Create accelerator node pool based on earlier selection
    case "$NODE_TYPE_CHOICE" in
        2)
            if [ -n "$GPU_TYPE" ]; then
                echo ""
                echo "Creating GPU node pool ($GPU_TYPE)..."
                if gcloud container node-pools create gpu-pool \
                    --cluster="$CLUSTER_NAME" \
                    --zone="$ZONE" \
                    --project="$PROJECT_ID" \
                    --machine-type="$GPU_MACHINE_TYPE" \
                    --accelerator "type=$GPU_TYPE,count=1" \
                    --num-nodes=1 \
                    --scopes=gke-default,storage-full; then
                    echo -e "${GREEN}✓ GPU node pool created${NC}"
                else
                    echo -e "${RED}✗ Failed to create GPU node pool${NC}"
                fi

                echo ""
                echo "Installing NVIDIA GPU device drivers..."
                kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
                echo -e "${GREEN}✓ GPU driver installation initiated${NC}"
            fi
            ;;
        3)
            if [ -n "$TPU_TOPOLOGY" ]; then
                echo ""
                echo "Creating TPU node pool ($TPU_POOL_NAME, topology: $TPU_TOPOLOGY)..."
                if gcloud container node-pools create "$TPU_POOL_NAME" \
                    --cluster="$CLUSTER_NAME" \
                    --zone="$ZONE" \
                    --project="$PROJECT_ID" \
                    --machine-type="$TPU_MACHINE_TYPE" \
                    --tpu-topology="$TPU_TOPOLOGY" \
                    --num-nodes="$TPU_NUM_NODES" \
                    --scopes=gke-default,storage-full; then
                    echo -e "${GREEN}✓ TPU node pool created${NC}"
                else
                    echo -e "${RED}✗ Failed to create TPU node pool${NC}"
                fi
            fi
            ;;
        *)
            echo -e "${GREEN}Using default CPU node pool.${NC}"
            ;;
    esac

    # Verify connection
    echo ""
    echo "Verifying cluster connection..."
    if kubectl get nodes &> /dev/null; then
        echo -e "${GREEN}✓ kubectl connected to cluster${NC}"
    else
        echo -e "${RED}✗ Failed to connect to cluster${NC}"
    fi
fi

# Setup Summary
echo ""
echo "=========================================="
echo "  Setup Summary"
echo "=========================================="
echo ""
echo -e "${GREEN}✓ Project:${NC} $PROJECT_ID"
echo -e "${GREEN}✓ Region:${NC} ${KERAS_REMOTE_ZONE:-us-central1-a (default)}"
echo -e "${GREEN}✓ Artifact Registry:${NC} $AR_LOCATION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME"
if [ "$SETUP_GKE" = true ]; then
    echo -e "${GREEN}✓ Backend:${NC} GKE"
    echo -e "${GREEN}✓ GKE Cluster:${NC} $CLUSTER_NAME"
fi
echo ""
echo "Important notes:"
echo "  1. Ensure your project has billing enabled"
echo "  2. Check your quotas for:"
echo "     - TPU/GPU availability in your region"
echo "     - Cloud Build concurrent builds"
if [ "$SETUP_GKE" = true ]; then
    echo "     - GKE node pools (GPU nodes require separate quota)"
fi
echo "  3. Add environment variables to your shell config:"
echo "     export KERAS_REMOTE_PROJECT=$PROJECT_ID"
echo "     export KERAS_REMOTE_ZONE=${KERAS_REMOTE_ZONE:-us-central1-a}"
if [ "$SETUP_GKE" = true ]; then
    echo "     export KERAS_REMOTE_CLUSTER=$CLUSTER_NAME  # optional, for custom cluster names"
fi
echo ""
echo "To view quotas:"
echo "  https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID"
echo ""
echo "To check TPU/GPU availability:"
echo "  gcloud compute accelerator-types list --filter=\"zone:$ZONE\""
echo ""
echo -e "${GREEN}Setup complete!${NC} You can now run:"
if [ "$SETUP_GKE" = true ]; then
    echo "  python example_gke.py"
fi
echo ""
